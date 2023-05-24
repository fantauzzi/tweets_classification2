from pathlib import Path
from shutil import copytree, rmtree

import hydra
import torch
import transformers
import wandb
from datasets import load_dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    DistilBertForSequenceClassification

from utils import info, warning, compute_metrics, get_eval_f1_from_best_epoch, log_model, log_nvidia_smi, setup_paths


def train(params: DictConfig) -> None:
    info(f'Current working directory is {Path.cwd()}')
    hydra_output_dir = OmegaConf.to_container(HydraConfig.get().runtime)['output_dir']
    info(f'Output dir is {hydra_output_dir}')

    ''' Set the RNG seed to make runs reproducible '''

    seed = params.transformers.get('seed')
    if seed is not None:
        transformers.set_seed(params.transformers.seed)

    (repo_root, models_path, tuned_model_path) = setup_paths(params)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        warning(f'No GPU found, device type is {device.type}')

    # Save the output of nvidia-smi (GPU info) into a text file, log it with MLFlow then delete the file

    emotions = load_dataset('emotion')  # num_proc=16
    pretrained_model = params.transformers.pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)

    emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

    num_labels = 6

    info(f'Training set contains {len(emotions_encoded["train"])} samples')
    model_name = f"{pretrained_model}-finetuned-emotion"

    with wandb.init(params.wandb.project, config={'params': OmegaConf.to_object(params)}) as run:
        if device.type == 'cuda':
            log_nvidia_smi(run)

        output_dir = str(models_path / model_name / 'fine-tuning')
        model: DistilBertForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=num_labels).to(device)
        """
        artifact = Artifact(name='pre-trained_model', type='URL')
        artifact.add_reference(uri='https://huggingface.co/distilbert-base-uncased')
        run.log_artifact(artifact)
        """

        training_args = TrainingArguments(output_dir=output_dir,
                                          load_best_model_at_end=True,
                                          **OmegaConf.to_object(params.training))  # dataloader_num_workers=16

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
        info(f'Model fine tuning results: {res}')

        if run.sweep_id is None:
            info(f'Saving fine tuned model in {tuned_model_path}')
            trainer.save_model(tuned_model_path)
            log_model(run=run, name='fine-tuned_model', local_path=tuned_model_path)
        else:
            api = wandb.Api()
            sweep_long_id = f'{run.entity}/{run.project}/{run.sweep_id}'
            sweep = api.sweep(sweep_long_id)
            best_run = sweep.best_run()
            eval_f1, best_loss, best_step = get_eval_f1_from_best_epoch(trainer.state.log_history)
            if best_run.id == run.id:
                info(f'Current trial (run) improved the evaluation metric to {eval_f1}')
                if Path(tuned_model_path).exists():
                    info(
                        f'Overwriting {tuned_model_path} with best fine tuned model so far, coming from checkpoint {trainer.state.best_model_checkpoint}')
                    rmtree(tuned_model_path)
                else:
                    info(
                        f'Saving best fine tuned model so far into {tuned_model_path}, coming from checkpoint {trainer.state.best_model_checkpoint}')
                copytree(trainer.state.best_model_checkpoint, tuned_model_path)


@hydra.main(version_base='1.3', config_path='../config', config_name='params')
def main(params: DictConfig) -> None:
    train(params)


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
Allow the option to resume from a previous sweep -> Done
Log the fine-tuned model with wandb as a model -> Done

What should actually be an artifact? Should the URL to the pre-trained model be an artifact? Perhaps a parameter instead

Version the choice of best model
Support the Netron viewer
Try setting the WANDB_DIR env variable https://docs.wandb.ai/guides/artifacts/storage
Implement proper validation and test
Reintroduce plot of confusion matrix
Make a GUI via gradio and / or streamlit
Version the saved model(also the dataset?)
Follow Andrej recipe 
Plot charts to W&B for debugging of the training process, as per Andrej's lectures
Give the model an API, deploy it, unit-test it
"""
