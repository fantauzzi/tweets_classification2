from pathlib import Path
from shutil import rmtree

import hydra
import numpy as np
import torch
import wandb
from datasets import load_dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from utils import info, warning, log_nvidia_smi, setup_paths, log_confusion_matrix


@hydra.main(version_base='1.3', config_path='../config', config_name='params')
def validate(params: DictConfig) -> None:
    info(f'Current working directory is {Path.cwd()}')
    hydra_output_dir = OmegaConf.to_container(HydraConfig.get().runtime)['output_dir']
    info(f'Output dir is {hydra_output_dir}')

    paths = setup_paths(params)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        warning(f'No GPU found, device type is {device.type}')

    emotions = load_dataset('emotion')  # num_proc=16

    with wandb.init(params.wandb.project,
                    notes='Validation and test of fine-tuned model',
                    dir = paths.wandb, config={'params': OmegaConf.to_object(params)}) as run:
        if device.type == 'cuda':
            log_nvidia_smi(run)

        if params.test is not None and params.test.model is not None:
            model_artifact = run.use_artifact(params.test.model)
            if paths.tuned_model.exists():
                rmtree(paths.tuned_model)
            info(f'Donwloading model {params.test.model} into {paths.tuned_model}')
            model_artifact.download(root=paths.models, recursive=True)

        model = AutoModelForSequenceClassification.from_pretrained(paths.tuned_model)
        tokenizer = AutoTokenizer.from_pretrained(paths.tuned_model)
        pipe = pipeline(model=model, task='text-classification', tokenizer=tokenizer, device=0)

        val_pred = pipe(emotions['validation']['text'])
        val_pred_labels = np.array([int(item['label'][-1]) for item in val_pred])
        y_val = np.array(emotions["validation"]["label"])
        eval_f1 = f1_score(y_val, val_pred_labels, average="weighted")
        eval_acc = accuracy_score(y_val, val_pred_labels)
        info(
            f'Validating inference pipeline with model loaded from {paths.tuned_model} - dataset contains {len(y_val)} samples')
        info(f'Validation f1 is {eval_f1} and validation accuracy is {eval_acc}')
        wandb.run.summary['test_f1'] = eval_f1
        wandb.run.summary['test_acc'] = eval_acc

        labels = emotions["train"].features["label"].names
        log_confusion_matrix('validation_confusion_matrix', val_pred_labels, y_val, labels, False)

        test_pred = pipe(emotions['test']['text'])
        test_pred_labels = np.array([int(item['label'][-1]) for item in test_pred])
        y_test = np.array(emotions["test"]["label"])
        test_f1 = f1_score(y_test, test_pred_labels, average="weighted")
        test_acc = accuracy_score(y_test, test_pred_labels)

        info(
            f'Testing inference pipeline with model loaded from {paths.tuned_model} - dataset contains {len(y_test)} samples')
        info(f'Test f1 is {test_f1} and test accuracy is {test_acc}')
        wandb.run.summary['test_f1'] = test_f1
        wandb.run.summary['test_acc'] = test_acc

        log_confusion_matrix('test_confusion_matrix', test_pred_labels, y_test, labels, False)

        info('Validation and test of the inference pipeline completed')


if __name__ == '__main__':
    validate()
