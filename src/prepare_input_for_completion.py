from pathlib import Path

import hydra
from datasets import load_dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from utils import info, setup_paths

# openai tools fine_tunes.prepare_data -f ../data/train.csv
# openai api fine_tunes.create -t "../data/train_prepared.jsonl" -v ../data/validation_prepared.jsonl  -m ada --compute_classification_metrics --classification_n_classes 6
# openai wandb sync <== should do it from the root of the repo
# openai api fine_tunes.results -i job-ID > ../data/report.csv

@hydra.main(version_base='1.3', config_path='../config', config_name='params')
def prepare_input(params: DictConfig) -> None:
    info(f'Current working directory is {Path.cwd()}')
    hydra_output_dir = OmegaConf.to_container(HydraConfig.get().runtime)['output_dir']
    info(f'Output dir is {hydra_output_dir}')
    paths = setup_paths(params)

    emotions = load_dataset('emotion')

    def save_csv(dataset: str) -> None:
        dataset_file = paths.data / f'{dataset}.csv'
        emotions[dataset].to_csv(dataset_file)
        tmp_file = paths.data / f'{dataset}.tmp.csv'
        with open(tmp_file, 'wt') as write_to:
            with open(dataset_file) as read_from:
                write_to.write('prompt,completion\n')
                read_from.readline()
                everything_else = read_from.readlines()
                write_to.writelines(everything_else)
        Path(dataset_file).unlink()
        Path(tmp_file).rename(str(dataset_file))
        info(f'Saved dataset with {len(emotions[dataset])} samples into {dataset_file}')

    save_csv('train')
    save_csv('validation')


if __name__ == '__main__':
    prepare_input()
