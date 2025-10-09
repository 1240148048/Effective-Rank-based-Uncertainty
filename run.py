"""Sample answers from LLMs on QA task."""
import gc
import os
import logging
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils
from uncertainty.uncertainty_measures import p_true as p_true_utils
from compute_uncertainty_measures import main as main_compute
from sklearn.metrics import auc
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from scipy.stats import entropy

os.environ["HTTP_PROXY"] = "http://10.37.1.23:12798"
os.environ["HTTPS_PROXY"] = "http://10.37.1.23:12798"

utils.setup_logger()

def er(embeddings):
    Hs = 0
    U, s, Vh = np.linalg.svd(embeddings.astype(np.float32), full_matrices=False)
    s = s.tolist()
    s_sum = sum(s)
    print(s[:10])
            
    for x in s:
        y = x / s_sum
        if y < 1e-8:
            break
        else:
            Hs -= y * math.log(y)
    return Hs
                    
def predict_with_embeddings(model, prompt, temperature=1, max_new_tokens=20):
    inputs = model.tokenizer(prompt, return_tensors="pt").to('cuda')
    input_ids = inputs.input_ids

    with torch.no_grad():
        generated_outputs = model.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            output_hidden_states=True, 
            output_scores=True, 
            return_dict_in_generate=True,
            do_sample=True if temperature > 0 else False
        )
        
        generated_ids = generated_outputs.sequences[0]
        
        new_token_ids = generated_ids[len(input_ids[0]):]
        
        predicted_answer = model.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        if '\n' in predicted_answer:
            predicted_answer = predicted_answer.split('\n')[0]
        
        token_log_likelihoods = []
        if hasattr(generated_outputs, 'scores') and generated_outputs.scores is not None:
            for i, score in enumerate(generated_outputs.scores):
                logits = score / temperature
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_likelihoods.append(log_probs[0, new_token_ids[i]].item())
        else:
            print("Warning: No scores available in generation output")
            token_log_likelihoods = [0.0] * len(new_token_ids)
        
        embeddings_per_layer = []
        if hasattr(generated_outputs, 'hidden_states') and generated_outputs.hidden_states is not None:
            last_step_hidden_states = generated_outputs.hidden_states[-1]
            for layer_idx in range(len(last_step_hidden_states)):

                embedding = last_step_hidden_states[layer_idx][0, -1, :].cpu().numpy()
                embeddings_per_layer.append(embedding)
            l_embedding = last_step_hidden_states[-1][0, -1, :].cpu().unsqueeze(0)
        else:
            print("Warning: No hidden states available in generation output")
            with torch.no_grad():
                outputs = model.model(**inputs, output_hidden_states=True)
                for layer_idx in range(len(outputs.hidden_states)):
                    embedding = outputs.hidden_states[layer_idx][0, -1, :].cpu().numpy()
                    embeddings_per_layer.append(embedding)
                l_embedding = outputs.hidden_states[-1][0, -1, :].cpu()
    
    return predicted_answer, token_log_likelihoods, embeddings_per_layer, l_embedding
    
def calculate_roc_auc(y_true, y_scores):
    thresholds = np.sort(np.unique(y_scores))[::-1]
    
    tprs = [0]
    fprs = [0]
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tprs.append(tpr)
        fprs.append(fpr)
    
    tprs.append(1)
    fprs.append(1)
    
    roc_auc = auc(fprs, tprs)
    
    return roc_auc, fprs, tprs

def eigenscore(A, alpha=0.001):
    cov_matrix = np.cov(A)
    
    eigenvalues = np.linalg.eigvalsh(cov_matrix).tolist()
    eigenscores = sum(math.log(eigenvalue+alpha) for eigenvalue in eigenvalues)
    
    return eigenscores
    
def main(args):

    # Setup run.
    if args.dataset == 'svamp':
        if not args.use_context:
            logging.info('Forcing `use_context=True` for svamp dataset.')
            args.use_context = True
    elif args.dataset == 'squad':
        if not args.answerable_only:
            logging.info('Forcing `answerable_only=True` for squad dataset.')
            args.answerable_only = True

    experiment_details = {'args': args}
    random.seed(args.random_seed)
    user = os.environ['USER']
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")

    wandb.init(
        entity=args.entity,
        project="semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug",
        dir=f"{scratch_dir}/{user}/uncertainty",
        config=args,
        notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
    )
    logging.info('Finished wandb init.')

    # Get accuracy metric.
    metric = utils.get_metric(args.metric)

    # Load dataset.
    train_dataset, validation_dataset = load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed)
    if args.ood_train_dataset is not None:
        logging.warning(
            'Using OOD dataset %s to construct few-shot prompts and train p_ik.',
            args.ood_train_dataset)
        # Get indices of answerable and unanswerable questions and construct prompt.
        train_dataset, _ = load_ds(args.ood_train_dataset, add_options=args.use_mc_options)
    if not isinstance(train_dataset, list):
        logging.info('Train dataset: %s', train_dataset)

    # Get indices of answerable and unanswerable questions and construct prompt.
    answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)

    if args.answerable_only:
        unanswerable_indices = []
        val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    experiment_details['prompt_indices'] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    # Create Few-Shot prompt.
    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, arg, make_prompt)
    experiment_details['prompt'] = prompt
    experiment_details['BRIEF'] = BRIEF
    logging.info('Prompt is: %s', prompt)

    # Initialize model.
    model = utils.init_model(args)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token="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")

    # Initialize prompt for p_true baseline.
    if args.compute_p_true:
        logging.info(80*'#')
        logging.info('Constructing few-shot prompt for p_true.')

        p_true_indices = random.sample(answerable_indices, args.p_true_num_fewshot)
        remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))
        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=model, dataset=train_dataset, indices=p_true_indices,
            prompt=prompt, brief=BRIEF,
            brief_always=args.brief_always and args.enable_brief,
            make_prompt=make_prompt, num_generations=args.num_generations,
            metric=metric)
        wandb.config.update(
            {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))
        experiment_details['p_true_indices'] = p_true_indices
        experiment_details['p_true_responses'] = p_true_responses
        experiment_details['p_true_few_shot_prompt'] = p_true_few_shot_prompt
        logging.info('Finished constructing few-shot prompt for p_true.')
        logging.info(80*'#')
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80*'#')

    # Start answer generation.
    logging.info(80 * '=')
    logging.info('Generating answers: ')
    logging.info(80 * '=')
    for dataset_split in ['train', 'validation']:
        logging.info(80 * 'x')
        logging.info('Starting with dataset_split %s.', dataset_split)
        logging.info(80 * 'x')

        # This will store all input data and model predictions.
        accuracies, generations, results_dict, p_trues = [], {}, {}, []
        all_cosine_similarities = []  # Store cosine similarities for all examples

        if dataset_split == 'train':
            if not args.get_training_set_generations:
                logging.info('Skip training data.')
                continue
            dataset = train_dataset
            possible_indices = list(set(remaining_answerable) | set(unanswerable_indices))

        else:
            dataset = validation_dataset
            possible_indices = range(0, len(dataset))

        # Evaluate over random subset of the datasets.
        indices = random.sample(possible_indices, min(args.num_samples, len(dataset)))
        experiment_details[dataset_split] = {'indices': indices}

        if args.num_samples > len(dataset):
            logging.warning('Not enough samples in dataset. Using all %d samples.', len(dataset))

        it = 0
        y_error=[]
        entropy_scores=[]
        entropy_scores1=[]
        entropy_scores5=[]
        entropy_scoresmid=[]
        Eigenscores = []
        
        for index in tqdm(indices):
            if (it + 1 % 10) == 0:
                gc.collect()
                torch.cuda.empty_cache()
            it += 1

            # Grab example at index.
            example = dataset[index]
            question, context = example["question"], example['context']
            generations[example['id']] = {'question': question, 'context': context}
            correct_answer = example['answers']['text']

            current_input = make_prompt(
                context, question, None, BRIEF, args.brief_always and args.enable_brief)
            local_prompt = prompt + current_input

            logging.info('Current input: '.ljust(15) + current_input)

            full_responses = []
            embeddings = []  # Store embeddings for all responses

            # We sample one low temperature answer on which we will compute the
            # accuracy and args.num_generation high temperature answers which will
            # be used to estimate the entropy variants.

            if dataset_split == 'train' and args.get_training_set_generations_most_likely_only:
                num_generations = 1
            else:
                num_generations = args.num_generations + 1

            ans_embed = {}
            for i in range(args.num_generations):
                temperature = 0.1 if i == 0 else args.temperature

                predicted_answer, token_log_likelihoods, embedding, l_embedding = predict_with_embeddings(
                    model, local_prompt, temperature)
                
                #embedding = embedding.cpu() if embedding is not None else None

                if ans_embed and predicted_answer.lower() in ans_embed.keys():
                    embedding = ans_embed[predicted_answer.lower()]
                else:
                    ans_embed[predicted_answer.lower()]=embedding
                    
                embeddings.append(embedding)

                # Only compute accuracy if question is answerable.
                compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
                if correct_answer and compute_acc:
                    acc = metric(predicted_answer, example, model)
                else:
                    acc = 0

                if i == 0:
                    y_error.append(1-acc)
                    logging.info('Iteration ' + str(it) + ':  ' + 80*'#')
                    if args.use_context:
                        logging.info('context: '.ljust(15) + str(context))
                    logging.info('question: '.ljust(15) + question)
                    logging.info('low-t prediction: '.ljust(15) + predicted_answer)
                    logging.info('correct answer: '.ljust(15) + str(correct_answer))
                    logging.info('accuracy: '.ljust(15) + str(acc))

                    accuracies.append(acc)
                    most_likely_answer_dict = {
                        'response': predicted_answer,
                        'token_log_likelihoods': token_log_likelihoods,
                        'embedding': l_embedding,
                        'accuracy': acc}
                    generations[example['id']].update({
                        'most_likely_answer': most_likely_answer_dict,
                        'reference': utils.get_reference(example)})

                else:
                    logging.info('high-t prediction '.ljust(15) + str(i) + ' : ' + predicted_answer)
                    # Aggregate predictions over num_generations.
                    full_responses.append(
                        (predicted_answer, token_log_likelihoods, embedding, acc))

            # Append all predictions for this example to `generations`.
            generations[example['id']]['responses'] = full_responses

            # Calculate cosine similarities between embeddings
            if len(embeddings) > 1 and embeddings[0] is not None:
                cosine_similarities = []
                first_embedding = embeddings[0]
                
                for i in range(1, len(embeddings)):
                    if embeddings[i] is not None:
                        # Calculate cosine similarity
                        cos_sim = F.cosine_similarity(
                            torch.tensor(first_embedding[-1]),#.unsqueeze(0), 
                            torch.tensor(embeddings[i][-1]),
                            dim=0
                        ).item()
                        cosine_similarities.append(min(1,cos_sim))
                    else:
                        cosine_similarities.append(None)
                
                generations[example['id']]['cosine_similarities'] = cosine_similarities
                all_cosine_similarities.extend([cs for cs in cosine_similarities if cs is not None])
                
                #logging.info(f'Cosine similarities with first response: {cosine_similarities}')
            else:
                generations[example['id']]['cosine_similarities'] = []
                logging.info('Not enough embeddings to calculate cosine similarities')

            embeddings = np.array(embeddings)
            embeddings1 = embeddings[:, -1, :]
            embeddings5 = embeddings[:, -5:, :].reshape(-1, embeddings.shape[-1])
            mid = (embeddings.shape[1]-1)//2
            es_em = embeddings[:, mid, :]
            embeddings = embeddings[:, mid-2 : mid+3, :].reshape(-1, embeddings.shape[-1])
            Hs = 0
            
            #U, s, Vh = np.linalg.svd(embeddings-np.mean(embeddings, axis=0), full_matrices=False)
            U, s, Vh = np.linalg.svd(embeddings.astype(np.float32), full_matrices=False)
            s = s.tolist()
            s_sum = sum(s)
            print(s[:10])
            
            for x in s:
                y = x / s_sum
                if y < 1e-8:
                    break
                else:
                    Hs -= y * math.log(y)

            entropy_scores.append(Hs)

            entropy_scores1.append(er(embeddings1))
            entropy_scores5.append(er(embeddings5))
            entropy_scoresmid.append(er(es_em))
            print(f"Hs: {entropy_scoresmid[-1]}")
            
            Eigenscores.append(eigenscore(es_em))
            print("eigenscore:", Eigenscores[-1])

            if args.compute_p_true and dataset_split == 'validation':
                # Already compute p_true here. Avoid cost of generations in compute_uncertainty script.
                p_true = p_true_utils.calculate_p_true(
                    model, question, most_likely_answer_dict['response'],
                    [r[0] for r in full_responses], p_true_few_shot_prompt,
                    hint=args.p_true_hint)
                p_trues.append(p_true)
                logging.info('p_true: %s', p_true)

        auc_roc, fprs, tprs = calculate_roc_auc(np.array(y_error), np.array(entropy_scores))
        auc_roc1, fprs, tprs = calculate_roc_auc(np.array(y_error), np.array(entropy_scores1))
        auc_roc5, fprs, tprs = calculate_roc_auc(np.array(y_error), np.array(entropy_scores5))
        auc_rocmid, fprs, tprs = calculate_roc_auc(np.array(y_error), np.array(entropy_scoresmid))
        print("AUC-ROC for our method =", auc_rocmid)
        print(f"-1: {auc_roc1}, -5: {auc_roc5}, mid5: {auc_roc}")

        auc_roc2, fprs2, tprs2 = calculate_roc_auc(np.array(y_error), np.array(Eigenscores))
        print("AUC-ROC for our Eigenscore =", auc_roc2)
        

        # Save generations for that split.
        utils.save(generations, f'{dataset_split}_generations.pkl')

        # Log overall accuracy and cosine similarity statistics
        accuracy = np.mean(accuracies)
        print(f"Overall {dataset_split} split accuracy: {accuracy}")
        wandb.log({f"{dataset_split}_accuracy": accuracy})
        
        if all_cosine_similarities:
            avg_cosine_sim = np.mean(all_cosine_similarities)
            std_cosine_sim = np.std(all_cosine_similarities)
            wandb.log({
                f"{dataset_split}_avg_cosine_similarity": avg_cosine_sim,
                f"{dataset_split}_std_cosine_similarity": std_cosine_sim
            })
            logging.info(f"Average cosine similarity for {dataset_split}: {avg_cosine_sim}")
            logging.info(f"Std cosine similarity for {dataset_split}: {std_cosine_sim}")

        if dataset_split == 'validation':
            if args.compute_p_true:
                results_dict['uncertainty_measures'] = {
                    'p_false':  [1 - p for p in p_trues],
                    'p_false_fixed':  [1 - np.exp(p) for p in p_trues],
                }
            utils.save(results_dict, 'uncertainty_measures.pkl')

    utils.save(experiment_details, 'experiment_details.pkl')
    logging.info('Run complete.')
    del model


if __name__ == '__main__':

    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    if args.compute_uncertainties:
        args.assign_new_wandb_id = False

    # First sample generations from LLM.
    logging.info('STARTING `generate_answers`!')
    main(args)
    logging.info('FINISHED `generate_answers`!')

    if args.compute_uncertainties:
        # Follow with uncertainty calculation script by default.
        args.assign_new_wandb_id = False
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(50 * '#X')
        logging.info('STARTING `compute_uncertainty_measures`!')
        main_compute(args)
        logging.info('FINISHED `compute_uncertainty_measures`!')