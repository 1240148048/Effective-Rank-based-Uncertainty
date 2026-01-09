# Revisiting Hallucination Detection Through The Lens Of Effective Ranks


## Installation Guide


To install Python with all necessary dependencies, we recommend the use of conda


```
conda-env update -f environment.yaml
conda activate effective_rank
export USER="root"
export HF_HOME=/root/autodl-tmp/huggingface
export WANDB_BASE_URL="https://api.bandw.top"
huggingface-cli login
```

For all tasks except BioASQ, the dataset is downloaded automatically from the Hugging Face Datasets library upon first execution, while BioASQ needs to be [downloaded](http://participants-area.bioasq.org/datasets) manually and stored at `$SCRATCH_DIR/$USER/code/data/bioasq/training11b.json`

Our experiments rely on [wandb](https://wandb.ai/) to log and save individual runs. While wandb will be installed automatically with the above conda script, you may need to log in with your wandb API key upon initial execution.

Our experiments rely on Hugging Face for all LLM models and most of the datasets. It may be necessary to set the environment variable HUGGING_FACE_HUB_TOKEN to the token associated with your Hugging Face account. Further, it may be necessary to apply for access to use the official repository of Meta's LLaMa-2 models.

## Demo

Execute

```
python run.py --model_name=Llama-2-7b-chat --dataset=trivia_qa
```

to reproduce results for LLaMa-2 Chat (7B) on the TriviaQA dataset. The baseline results will also be provided after all responses have been generated.


## Further Instructions


### Repository Structure

We here give an overview of the various components of the code.

By default, a standard run executes the following three scripts in order:

* `run.py`: Sample responses (and their likelihods/hidden states) from the models for a set of input questions.
* `compute_uncertainty_measures.py`: Compute uncertainty metrics given responses.
* `analyze_results.py`: Compute aggregate performance metrics given uncertainties.

## Additional Experiments

### Modification of Evaluation Logic

We conduct additional experiments on CoQA, MATH-500, HumanEval, and CNN/DailyMail.As the tasks in these datasets are more complex and the answer space is more open-ended, ROUGE-L is no longer an appropriate metric for evaluating answer correctness.

Instead, the evaluation strategy is changed to an LLM-based metric, where a large language model is prompted to judge the correctness of the generated answer. The code can be executed as follows:

```
python run.py --model_name=Llama-2-7b-chat --dataset=coqa --metric llm
```

The original implementation uses an additional entailment model for evaluation.
If one wishes to let the same LLM used for generation (or another powerful model such as GPT-4) evaluate the correctness of the answer during the analysis stage, modifications should be made to: ***/uncertainty/utils/utils.py (lines 199–244)***

### Different datasets require dataset-specific evaluation criteria:

- CoQA: As a multi-turn dialogue dataset, the evaluation must consider previous conversational context together with the ground-truth answer to determine correctness.

- HumanEval: The evaluation should judge whether the generated code is logically correct, runnable, and fulfills the programming requirements, based on both the reference solution and the problem description.

- CNN/DailyMail: The evaluation focuses on whether the generated summary captures the main ideas of the reference summary, rather than exact wording overlap.

- MATH-500 (special handling): The model is required to output the final answer in the following format: **#### \{answer\}**, where no chain-of-thought is included. If the answer is a decimal, it should be rounded to two decimal places.

All prompts and implementation details for these evaluation strategies can be found in ***prompt.md*.**

### Notes on Fairness and Efficiency

After modifying the evaluation functions, it is important to ensure fairness across different methods, i.e., all approaches should be evaluated under consistent standards.

Additionally, the maximum token generation length should be adjusted according to each dataset to ensure complete output generation, and reduce unnecessary computational overhead.

### Dataset Loading

All datasets are publicly available on open-source platforms such as Hugging Face.

To enable their usage, please add the corresponding --dataset_name options (i.e., the new dataset names) and their loading paths in: ***/uncertainty/data/data_utils.py***. This process can be efficiently implemented using vibe coding.

## Acknowledgements

Our implementation of semantic entropy calculation is built upon and inspired by the codebase from [jlko/semantic_uncertainty](https://github.com/jlko/semantic_uncertainty). We thank the authors for making their code publicly available.

We also acknowledge the foundational work presented in the paper **"Detecting Hallucinations in Large Language Models Using Semantic Entropy"**. Their formulation of semantic entropy provided significant inspiration for our approach to quantifying uncertainty in model generations.

While our work adapts and integrates these components into a broader framework for a different application, we are grateful for the clarity and quality of the original research and code, which greatly facilitated our development process.
