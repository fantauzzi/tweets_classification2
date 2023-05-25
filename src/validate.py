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
    # pretrained_model = params.transformers.pretrained_model
    # tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    """def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)"""

    with wandb.init(params.wandb.project, dir = paths.wandb, config={'params': OmegaConf.to_object(params)}) as run:
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

"""
TODO Add a test step -> Done
Turn the  script into an MLFlow project -> Done
Store hyperparameters in a config.file -> Done
Introduce proper logging -> Done
Tune hyperparameters(using some framework / library) -> Done
(Re -)train and save -> Done
Do proper testing of inference from saved model -> Done
Draw charts of the training and validation loss and the confusion matrix under MLFlow -> Done
Implement early-stopping -> Done
Make a nested run for every Optuna trial -> Done
Send the training to the cloud -> Done
Make sure GPU info is logged -> Done
Ensure parameters for every trial are logged, at least the changing ones -> Done
Split MLFlowTrialCB() in its own file -> Done
Try GPU on Amazon/google free service -> Done
Have actually random run names even with a set random seed -> Done
Fix reproducibility -> Done
Make sure hyperparameters search works correctly -> Done
Can fine-tuning be interrupted and resumed? -> Done, yes!
Fix up call to optimize() -> Done
Optimize hyper-parameters tuning such that it saves the best model so far at every trial, so it doesn't have to be
    computed again later (is it even possible?) -> Done (yes it is)
Provide an easy way to coordinate the trial info (in the SQLite DB) with the run info in MLFlow -> Done
Log with MLFlow the Optuna trial id of every nested run, also make sure the study name is logged -> Done
Allow the option to resume from a previous sweep -> Done

What should actually be an artifact? Should the URL to the pre-trained model be an artifact? Perhaps a parameter instead
Log the fine-tuned model with wandb as a model

Support the Netron viewer
Try setting the WANDB_DIR env variable https://docs.wandb.ai/guides/artifacts/storage
Version the choice of best model
Implement proper validation and test
Reintroduce plot of confusion matrix
Make a GUI via gradio and / or streamlit
Version the saved model(also the dataset?)
Follow Andrej recipe 
Plot charts to W&B for debugging of the training process, as per Andrej's lectures
Give the model an API, deploy it, unit-test it
"""
