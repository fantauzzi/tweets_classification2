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

    dataset_file = paths.data / 'validation_set_prepared.jsonl'
    emotions['train'].to_csv(dataset_file)
    if Path(dataset_file).exists():
        info(f'Output file {dataset_file} exists and will be overwritten')
    suffix = " ->"
    info(
        f'Saving validation set with {len(dataset)} samples prepared for fine-tuned model {params.openai.fine_tuned_model} in {dataset_file}')
    with open(dataset_file, 'wt') as jsonl:
        for user_message in tqdm(dataset):
            line = {'prompt': f'{user_message}{suffix}',
                    'max_tokens': 1,
                    'logprobs': 6,
                    'model': params.openai.fine_tuned_model}
            jsonl_line = json.dumps(line) + '\n'
            jsonl.write(jsonl_line)
    sleep(.1)  # Allow tqdm to make the last update to its progress bar


if __name__ == '__main__':
    prepare_input()
