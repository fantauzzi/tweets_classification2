import json

import hydra
import numpy as np
import wandb
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score

from utils import setup_paths, info, log_confusion_matrix, error, warning


@hydra.main(version_base='1.3', config_path='../config', config_name='params')
def validate(params: DictConfig) -> None:
    paths = setup_paths(params)
    results_file = params.openai.results_file
    results_file_path = paths.data / results_file
    info(f"Testing model with inferences loaded from {results_file_path}")
    with open(results_file_path) as jsonl:
        entries = jsonl.readlines()
    info(f"Loaded {len(entries)} inferences from file")
    entries_json = [json.loads(entry) for entry in entries]

    emotions = load_dataset('emotion')

    labels = emotions["test"].features["label"].names
    n_samples = len(emotions['test'])
    n_parsed = 0
    n_unparsable = 0
    model = None
    predictions = np.full(shape = (n_samples,), fill_value=-1, dtype = int)
    for entry in entries_json:
        try:
            if model is None:
                model = entry[1]['model']
            else:
                assert model == entry[1]['model']
            pred = int(entry[1]['choices'][0]['text'][-1])
            assert 0 <= pred
            assert pred <= 5
            idx = int(entry[2]['row_id'])
            assert predictions[idx] == -1
            predictions[idx]=pred
        except Exception as ex:
            warning(f'Cannot parse inference because of exception {repr(ex)}')
            warning(f'  Offending inference: {entry}')
            n_unparsable += 1
        n_parsed += 1

    if n_unparsable > 0:
        warning(f"Found {n_unparsable} inferences that couldn't be parsed")
    else:
        info('All inferences have been parsed correctly')
    if n_parsed != n_samples:
        error(f'Expected {n_samples} inferences but have been able to parse {n_parsed} instead')

    with wandb.init(project=params.wandb.project,
                    notes='Test of fine-tuned model',
                    dir=paths.wandb,
                    config={'params': OmegaConf.to_object(params)}) as run:

        y_test = np.array(emotions["test"]["label"])
        test_f1 = f1_score(y_test, predictions, average="weighted")
        test_acc = accuracy_score(y_test, predictions)
        info(f'Validation f1 is {test_f1} and validation accuracy is {test_acc}')
        wandb.run.summary['test_f1'] = test_f1
        wandb.run.summary['test_acc'] = test_acc

        log_confusion_matrix('test_confusion_matrix', predictions, y_test, labels, False)


if __name__ == '__main__':
    validate()


