import logging
from pathlib import Path

import numpy as np
import transformers
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from wandb.sdk.wandb_run import Run

from wandb import Artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
_logger = logging.getLogger()
info = _logger.info
warning = _logger.warning


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


def plot_confusion_matrix(y_preds, y_true, labels, show=True):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    if show:
        plt.show()
    return fig


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


def log_model_as_artifact(run: Run, name: str, local_path: str | Path) -> Artifact:
    artifact = Artifact(name=name, type='model')
    artifact.add_dir(local_path=local_path, name=Path(local_path).name)
    run.log_artifact(artifact)
    return artifact
