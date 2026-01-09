## Prompt for CoQA / HumanEval / CNN-DailyMail

```python
	prompt = (
        f"We are assessing the quality of answers to the following question:\n"
        f"{example['question']}\n"
    )

	if dataset_name.lower() == "coqa":
    	if "history" in example and example["history"]:
        	prompt += (
           		"This question is part of a multi-turn dialogue. "
            	"The previous conversation context is provided below:\n"
            	f"{example['history']}\n"
       		)

    	prompt += (
        	"When evaluating the proposed answer, you must consider:\n"
        	"- the full conversational context,\n"
        	"- the current question,\n"
        	"- and the ground-truth answer.\n"
        	"An answer is correct if it is contextually appropriate and "
        	"semantically equivalent to the ground-truth answer.\n\n"
    	)

	elif dataset_name.lower() == "humaneval":
    	if "prompt" in example:
        	prompt += (
            	"This is a programming problem. The problem description is:\n"
            	f"{example['prompt']}\n"
        	)

    	prompt += (
        	"When evaluating the proposed answer, you must determine whether:\n"
        	"- the generated code is logically correct,\n"
        	"- the code is syntactically valid and runnable,\n"
        	"- and the code fulfills the programming requirements described above.\n"
        	"You should judge correctness based on functional behavior, "
        	"not superficial similarity or formatting.\n\n"
    	)

	elif dataset_name.lower() in ["cnn_dailymail", "cnn", "cnndailymail"]:
    	if "article" in example:
        	prompt += (
            	"This is a news summarization task. The source article is:\n"
            	f"{example['article']}\n"
        	)

    	prompt += (
        	"When evaluating the proposed answer, focus on whether the summary:\n"
        	"- captures the main ideas and key information of the reference summary,\n"
        	"- preserves the core meaning of the original content.\n"
        	"Do NOT penalize differences in wording, phrasing, or sentence structure "
        	"as long as the main ideas are correctly conveyed.\n\n"
    	)

	if len(correct_answers) == 1:
    	prompt += f"The expected answer is:\n{correct_answers[0]}\n"
	else:
    	prompt += (
        	"The following are expected answers to this question:\n"
        	f"{correct_answers}\n"
    	)

	prompt += f"The proposed answer is:\n{predicted_answer}\n\n"

	if len(correct_answers) == 1:
        prompt += (
        	"Within the context of the task and the evaluation criteria above, "
        	"does the proposed answer mean the same as the expected answer?"
    	)
	else:
    	prompt += (
        	"Within the context of the task and the evaluation criteria above, "
        	"does the proposed answer mean the same as any of the expected answers?"
    	)

	prompt += " Respond only with yes or no.\nResponse:"
```

## Prompt for MATH-500 Final Answer Format

```
You must output ONLY the final numerical answer in the following format:

#### {answer}

Do NOT include any chain-of-thought or explanations.

Rules:
- The output must consist of exactly one line.
- {answer} must be a number.
- If the answer is a decimal, round it to TWO decimal places.
- If the answer is an integer, output it directly.
- Do not include units, symbols, or additional text.

Any output that does not strictly follow this format will be considered incorrect.
```

## Python Function for MATH-500 Answer Evaluation

```python
import re
from fractions import Fraction

def evaluate_math500_answer(gt_answer: str, model_output: str, tol: float = 0.01) -> bool:
	match = re.search(r"####\s*([-+]?\d*\.?\d+)", model_output)
	if match is None:
    	return False

	try:
    	pred_value = float(match.group(1))
	except ValueError:
    	return False
    
	try:
   		gt_answer = gt_answer.strip()
    	if "/" in gt_answer:
        	gt_value = float(Fraction(gt_answer))
    	else:
        	gt_value = float(gt_answer)
	except (ValueError, ZeroDivisionError):
    	return False

	return abs(pred_value - gt_value) < tol
```