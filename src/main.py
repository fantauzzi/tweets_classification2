from os import system
from pathlib import Path

import hydra
import mlflow as mf
import torch
import transformers
from datasets import load_dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    DistilBertForSequenceClassification

import wandb
from utils import info, warning, info_active_run, compute_metrics, get_name_for_run


@hydra.main(version_base='1.3', config_path='../config', config_name='params')
def main(params: DictConfig) -> None:
    """
    Tune the hyperparameters for best fine-tuning the model
    :param params: the configuration parameters passed by Hydra
    """

    ''' Set-up Hydra '''
    info(f'Current working directory is {Path.cwd()}')
    hydra_output_dir = OmegaConf.to_container(HydraConfig.get().runtime)['output_dir']
    info(f'Output dir is {hydra_output_dir}')

    ''' Set the RNG seed to make runs reproducible '''

    """
    seed = params.transformers.get('seed')
    if seed is not None:
        transformers.set_seed(params.transformers.seed)
    """

    ''' Set various paths '''

    # Absolute path to the repo root in the local filesystem
    repo_root = Path('..').resolve()

    # Absolute path to the MLFlow tracking dir; currently supported only in the local filesystem
    tracking_uri = repo_root / params.mlflow.tracking_uri
    mf.set_tracking_uri(tracking_uri)  # set_tracking_uri() expects an absolute path

    # Absolute path to the directory where model and model checkpoints are to be saved and loaded from
    models_path = (repo_root / params.main.models_dir).resolve()

    # Absolute path the fine-tuned model is saved to/loaded from
    tuned_model_path = models_path / params.main.fine_tuned_model_dir

    # Absolute path where the hyperparameter values for the fine-tuned model are saved to/loaded from
    params_override_path = models_path / 'params_override.yaml'

    ''' If there is no MLFlow run currently ongoing, then start one. Note that is this script has been started from
     shell with `mlflow run` then a run is ongoing already, no need to start it'''

    if mf.active_run() is None:
        info('No active MLFlow run, starting one now')
        mf.start_run(run_name=get_name_for_run())

    info_active_run()

    if not models_path.exists():
        models_path.mkdir()

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        warning(f'No GPU found, device type is {device.type}')

    # Save the output of nvidia-smi (GPU info) into a text file, log it with MLFlow then delete the file
    nvidia_info_filename = 'nvidia-smi.txt'
    nvidia_info_path = repo_root / nvidia_info_filename
    system(f'nvidia-smi -q > {nvidia_info_path}')
    mf.log_artifact(str(nvidia_info_path))
    nvidia_info_path.unlink(missing_ok=True)

    emotions = load_dataset('emotion')
    pretrained_model = params.transformers.pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)

    emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

    num_labels = 6

    info(f'Training set contains {len(emotions_encoded["train"])} samples')
    model_name = f"{pretrained_model}-finetuned-emotion"

    with wandb.init(params.wandb.project, config={'params': OmegaConf.to_object(params)}) as run:
        output_dir = str(models_path / model_name / 'fine-tuning')
        model: DistilBertForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=num_labels).to(device)
        training_args = TrainingArguments(output_dir=output_dir,
                                          load_best_model_at_end=True,
                                          **OmegaConf.to_object(params.training))

        callbacks = [
            transformers.EarlyStoppingCallback(
                early_stopping_patience=params.early_stopping.patience)] if params.early_stopping.patience > 0 else []

        trainer = Trainer(model=model,
                          args=training_args,
                          compute_metrics=compute_metrics,
                          train_dataset=emotions_encoded["train"],
                          eval_dataset=emotions_encoded["validation"],
                          tokenizer=tokenizer,
                          callbacks=callbacks)

        res = trainer.train()
        info(f'Model tuning completed with results {res}')

        info(f'Mode fine tuning results: {res}')
        trainer.save_model(tuned_model_path)


if __name__ == '__main__':
    main()

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

Tag the best nested run as such, will have to remove and re-assign the tag of best nested run as needed 
Log computation times
Make a GUI via gradio and / or streamlit
Version the saved model(also the dataset?)
Follow Andrej recipe
Plot charts to MLFlow for debugging of the training process, as per Andrej's lectures
Give the model an API, deploy it, unit-test it
"""
