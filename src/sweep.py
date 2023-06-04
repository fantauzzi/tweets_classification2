import os
from functools import partial
from os import system

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from train import train
from utils import info, setup_paths, log_model

'''
NOTE 
To start the sweep from command line, cd into the `src` directory and then:
wandb sweep --project <wandb_project> ../config/sweep.yaml

To set the `wandb` directoty location, set the `WANDB_DIR` env. variable before running the above

To resume a sweep (cannot be in Finished state):
wandb sweep --resume  <entity>/<wandb_project>/<sweep_id>
'''


@hydra.main(version_base='1.3', config_path='../config', config_name='params')
def main(params: DictConfig) -> None:
    (repo_root, _, tuned_model_path, wandb_dir, _) = setup_paths(params)
    config_path = (repo_root / 'config').resolve()
    sweep_config_path = config_path / 'sweep.yaml'

    if os.environ.get('WANDB_DIR') is None:
        os.environ['WANDB_DIR'] = str(wandb_dir)

    info(f'These are the parameter(s) set at the beginning of the sweep:')
    for key, value in params.items():
        info(f"  '{key}': {value}")

    # Configure the sweep

    info(f'Loading sweep configuration from {sweep_config_path}')
    info(f'Loading sweep configuration from {sweep_config_path}')
    sweep_configuration = OmegaConf.load(sweep_config_path)
    # If the sweep config. contains a 'command' key, then remove it. That is needed only to start a sweep from the CLI;
    # here it would generate an error: `omegaconf.errors.InterpolationKeyError: Interpolation key 'env' not found`
    if sweep_configuration.get('command') is not None:
        del sweep_configuration['command']
    sweep_configuration = OmegaConf.to_object(sweep_configuration)

    # Start the sweep

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=params.wandb.project)
    info(f'Starting sweep for {params.wandb.count} iteration(s) with id {sweep_id}')
    train_with_params = partial(train, params)
    wandb.agent(sweep_id, function=train_with_params, count=params.wandb.count)
    # Start a run in order to log the best model trained during the sweep
    info(f'Starting run to log the best optimzed model from the sweep, saved in {tuned_model_path}')
    with wandb.init(project=params.wandb.project,
                    notes='Logging of the best fine-tuned model produced during the sweep',
                    config={'params': OmegaConf.to_object(params)}) as run:
        log_model(run=run, name='fine-tuned_model', local_path=tuned_model_path)
        wandb.log(data={'sweep_id': sweep_id})

    # Stop the sweep

    api = wandb.Api()
    prj = api.project(params.wandb.project)
    sweep_long_id = f'{prj.entity}/{params.wandb.project}/{sweep_id}'
    command = f'wandb sweep --stop {sweep_long_id}'
    info(f'Stopping the current sweep {sweep_id} with command:')
    info(f'  {command}')
    system(command)


if __name__ == '__main__':
    main()
