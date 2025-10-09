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

## Acknowledgements

Our implementation of semantic entropy calculation is built upon and inspired by the codebase from [jlko/semantic_uncertainty](https://github.com/jlko/semantic_uncertainty). We thank the authors for making their code publicly available.

We also acknowledge the foundational work presented in the paper **"Detecting Hallucinations in Large Language Models Using Semantic Entropy"**. Their formulation of semantic entropy provided significant inspiration for our approach to quantifying uncertainty in model generations.

While our work adapts and integrates these components into a broader framework for a different application, we are grateful for the clarity and quality of the original research and code, which greatly facilitated our development process.
