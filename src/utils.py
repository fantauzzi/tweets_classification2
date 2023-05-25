import logging
from collections import namedtuple
from os import system
from pathlib import Path

import numpy as np
import transformers
import wandb
from matplotlib import pyplot as plt
from matplotlib.pyplot import Figure, get_backend
from omegaconf import DictConfig
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from wandb.sdk.wandb_run import Run

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
_logger = logging.getLogger()
info = _logger.info
warning = _logger.warning

info(f'Matplotlib backend is {get_backend()}')


def xor(a: bool, b: bool) -> bool:
    return (a and not b) or (b and not a)


def implies(a: bool, b: bool) -> bool:
    return not a or b


def compute_metrics(pred: transformers.trainer_utils.EvalPrediction) -> dict[str, np.float64]:
    ground_truth = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(ground_truth, preds, average="weighted")
    acc = accuracy_score(ground_truth, preds)
    return {"accuracy": acc, "f1": f1}


def plot_confusion_matrix(y_preds: np.ndarray, y_true: np.ndarray, labels: list[str], show=True) -> Figure:
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    if show:
        plt.show()
    return fig


def log_confusion_matrix(media_title: str, y_preds: np.ndarray, y_true: np.ndarray, labels: list[str],
                         show=True) -> None:
    fig_val = plot_confusion_matrix(y_preds, y_true, labels, show)
    conf_matrix_filename = 'confusion_matrix.png'
    try:
        fig_val.savefig(fname=conf_matrix_filename)
        wandb.log({media_title: wandb.Image(conf_matrix_filename)})
    finally:
        Path(conf_matrix_filename).unlink(missing_ok=True)


def get_eval_f1_from_best_epoch(log_history: list[dict]) -> (float, float, int):
    """
    Returns evaluation F1, evaluation loss and step number for the step with the lowest evaluation loss in a given
    Trainer.state.log_history
    :param log_history: the given Trainer.state.log_history
    :return: a triple with, in this order, the required F1 score, loss and step number
    """
    lowest_eval_loss = None
    step = None
    eval_f1 = None
    for entry in log_history:
        eval_loss = entry.get('eval_loss')
        if eval_loss is not None and (lowest_eval_loss is None or eval_loss < lowest_eval_loss):
            lowest_eval_loss = eval_loss
            step = entry['step']
            eval_f1 = entry['eval_f1']
    return eval_f1, lowest_eval_loss, step


def log_model(run: Run, name: str, local_path: str | Path) -> wandb.Artifact:
    artifact = wandb.Artifact(name=name, type='model')
    artifact.add_dir(local_path=local_path, name=Path(local_path).name)
    run.log_artifact(artifact)
    run.link_artifact(artifact=artifact, target_path='model-registry')
    return artifact


def log_nvidia_smi(run: Run) -> None:
    nvidia_info_filename = Path('nvidia-smi.txt')
    # nvidia_info_path = repo_root / nvidia_info_filename
    system(f'nvidia-smi -q > {str(nvidia_info_filename)}')
    artifact = wandb.Artifact(name='nvidia-smi', type='text')
    artifact.add_file(local_path=nvidia_info_filename)
    run.log_artifact(artifact)
    nvidia_info_filename.unlink(missing_ok=True)


Paths = namedtuple('Paths', ['repo_root', 'models', 'tuned_model', 'wandb'])


def setup_paths(params: DictConfig) -> Paths:
    # Absolute path to the repo root in the local filesystem
    repo_root = Path('..').resolve()

    # Absolute path to the directory where model and model checkpoints are to be saved and loaded from
    models_path = (repo_root / params.main.models_dir).resolve()

    # Absolute path the fine-tuned model is saved to/loaded from
    tuned_model_path = models_path / params.main.fine_tuned_model_dir

    if not models_path.exists():
        models_path.mkdir()

    wandb_path = repo_root / 'wandb'

    res = Paths(repo_root=repo_root, models=models_path, tuned_model=tuned_model_path, wandb=wandb_path)
    return res
