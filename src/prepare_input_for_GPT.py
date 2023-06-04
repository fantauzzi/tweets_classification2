import json
from pathlib import Path
from time import sleep

import hydra
from datasets import load_dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from utils import info, setup_paths


@hydra.main(version_base='1.3', config_path='../config', config_name='params')
def prepare_input(params: DictConfig) -> None:
    info(f'Current working directory is {Path.cwd()}')
    hydra_output_dir = OmegaConf.to_container(HydraConfig.get().runtime)['output_dir']
    info(f'Output dir is {hydra_output_dir}')
    paths = setup_paths(params)

    emotions = load_dataset('emotion')
    dataset = emotions['validation']['text']
    dataset_file = paths.data / params.openai.dataset_file
    with open(dataset_file, 'wt') as jsonl:
        for user_message in tqdm(dataset):
            delimiter = "####"
            system_message = f"""
            You will be provided with the text of tweets. \
            The tweets will be delimited with \
            {delimiter} characters. \
            Classify each tweet into a sentiment. \ 
            Provide your output in json format with the \
            keys: sentiment.
            sentiments: sadness, joy, love, anger, fear , surprise
            """
            messages = [
                {'role': 'system',
                 'content': system_message},
                {'role': 'user',
                 'content': f"{delimiter}{user_message}{delimiter}"},
            ]
            jsonl_line = f'{{"model": "gpt-3.5-turbo", "messages": {json.dumps(messages)},"temperature": {params.openai.temperature}}}\n'
            jsonl.write(jsonl_line)
    sleep(.1)  # Allow tqdm to make the last update to its progress bar
    info(f'Feteched {len(dataset)} predictions from API and saving them into {dataset_file}')


if __name__ == '__main__':
    prepare_input()
