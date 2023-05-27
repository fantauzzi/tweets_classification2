from shutil import rmtree

import numpy as np
import torch
from hydra import initialize, compose
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from wandb import Api

from utils import setup_paths, info, warning

initialize(version_base='1.3', config_path='../config')
params = compose(config_name='params')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != 'cuda':
    warning(f'No GPU found, device type is {device.type}')

paths = setup_paths(params)
if params.test is not None and params.test.model is not None:
    api = Api()
    artifact_long_name = f'{params.wandb.project}/{params.test.model}'
    model_artifact = api.artifact(name=artifact_long_name)
    if paths.tuned_model.exists():
        rmtree(paths.tuned_model)
    info(f'Donwloading model {params.test.model} into {paths.tuned_model}')
    # TODO replace paths.models with a computation of the needed path from params.tuned_model
    model_artifact.download(root=paths.models, recursive=True)

model = AutoModelForSequenceClassification.from_pretrained(paths.tuned_model).to(device)
tokenizer = AutoTokenizer.from_pretrained(paths.tuned_model)
pipe = pipeline(model=model, task='text-classification', tokenizer=tokenizer, device=0)


def predict(tweets: list[str]):
    val_pred = pipe(tweets)
    # val_pred_labels = np.array([int(item['label'][-1]) for item in val_pred])
    return val_pred


def main():
    tweets = [None, None]
    tweets[
        0] = "This is such a lovely day of double rainbows and unicorns and lollypops growing on trees, I am as happy as a pig in fertilizer"
    tweets[
        1] = "The thunderstorm is scaring the living thing out of my chihuahua and myself: we are hiding together under the table"
    res = predict(tweets)
    print(res)


if __name__ == '__main__':
    main()
