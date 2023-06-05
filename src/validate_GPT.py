import json

import hydra
import numpy as np
import wandb
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score

from utils import setup_paths, info, log_confusion_matrix


@hydra.main(version_base='1.3', config_path='../config', config_name='params')
def validate(params: DictConfig) -> None:
    paths = setup_paths(params)
    results_file = params.openai.results_file
    results_file_path = paths.data / results_file
    info(f"Validating GPT model with inferences loaded from {results_file_path}")
    with open(results_file_path) as jsonl:
        entries = jsonl.readlines()
    info(f"Loaded {len(entries)} inferences from file")
    entries_json = [json.loads(entry) for entry in entries]

    emotions = load_dataset('emotion')

    predictions = []
    labels = emotions["train"].features["label"].names
    labels_to_numbers = {label: i for i, label in zip(range(len(labels)), labels)}
    line = 1
    unparsable = 0
    for entry in entries_json:
        try:
            pred = json.loads(entry[1]['choices'][0]['message']['content'])['sentiment']
            predictions.append(labels_to_numbers[pred])
        except Exception as ex:
            info(f'Cannot parse GPT inference at line {line} because of exception {repr(ex)}')
            info(f"{entry[1]['choices'][0]['message']['content']}")
            predictions.append(len(labels))
            unparsable += 1
        line += 1

    info(f"Found {unparsable} GPT inferences that couldn't be parsed")

    with wandb.init(project=params.wandb.project,
                    notes='Validation and test of fine-tuned model',
                    dir=paths.wandb,
                    config={'params': OmegaConf.to_object(params)}) as run:

        val_pred_labels = np.array(predictions)
        y_val = np.array(emotions["validation"]["label"])
        eval_f1 = f1_score(y_val, val_pred_labels, average="weighted")
        eval_acc = accuracy_score(y_val, val_pred_labels)
        info(f'Validation f1 is {eval_f1} and validation accuracy is {eval_acc}')
        wandb.run.summary['eval_f1'] = eval_f1
        wandb.run.summary['eval_acc'] = eval_acc

        # log_confusion_matrix('validation_confusion_matrix', val_pred_labels, y_val, labels, False)


if __name__ == '__main__':
    validate()


""" 
TODO
====
Log GPT prompt with wandb
Find out about classification specific API with GPT and its fine-tuning
"""