#!/usr/bin/env python
from typing import Dict, Any
import copy
import os
import torch
import argparse
from tabluence.utilities.argument_parsing.train import get_train_parser
import tabluence.deep_learning.data.dataset
import tabluence.deep_learning.pipeline.training
import tabluence.deep_learning.pipeline.model
import tabluence.deep_learning.data.pipeline.pipeline
import tabluence.deep_learning.data.tensorizer
from tabluence.utilities.device import get_device
from tabluence.utilities.io.files_and_folders import clean_folder
from tabluence.utilities.randomization.seed import fix_random_seeds
from tabluence.contrib.mmcv import Config
import logging


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(args: argparse.Namespace) -> None:
    """
    Parameters
    ----------
    args: `argparse.Namespace`, required
        The arguments parsed from the command line.
    """
    # - fixing random seed
    fix_random_seeds(seed=args.seed)

    # - getting device
    device = get_device(device=args.device)

    # - getting the configuration
    config = dict(Config.fromfile(args.config))
    if args.clean:
        clean_folder(folder_path=config['trainer']['config']['checkpointing']['repo'])

    # - getting the dataloaders
    logger.info("~> preparing the dataloaders...\n")
    dataloaders_original = getattr(
        tabluence.deep_learning.data.dataset,
        config['data']['interface']
    )(**config['data']['args'])

    config['model']['config']['task']['label_layout'] = dataloaders_original['train'].dataset.label_layouts[config['model']['config']['task']['target_in_meta']]
    config['trainer']['config']['task'] = config['model']['config']['task']

    # - dumping config and args
    checkpointing_repo = config['trainer']['config']['checkpointing']['repo']
    os.makedirs(checkpointing_repo, exist_ok=True)
    torch.save({'config', 'args'}, os.path.join(checkpointing_repo, f'config_and_args.pt'))

    # - preparing the dataside pipeline
    logger.info("~> preparing the data-side processing pipeline...\n")
    dataside_pipeline = getattr(tabluence.deep_learning.data.pipeline.pipeline, config['dataside_pipeline']['type'])(
        train_dataloader=dataloaders_original['train'], **config['dataside_pipeline']['args']
    )

    # - updating the dataloaders:
    logger.info(f"~> preparing the dataloaders (modes: {[e for e in dataloaders_original.keys()]})...\n")
    dataloaders = {x: dataside_pipeline(dataloaders_original[x], mode=x) for x in dataloaders_original}

    # - preparing the tensorizer
    logger.info(f"~> preparing the tensorizer (tensorizer type: {config['tensorizer']['type']})...\n")
    tensorizer = getattr(tabluence.deep_learning.data.tensorizer, config['tensorizer']['type'])(
        config=config['tensorizer']['config'],
        device=device
    )

    logger.info(f"\t~>learning and initializing tensorizer...\n")
    tensorizer.learn(dataloader=dataloaders['train'])
    tensorizer = tensorizer.to(device)

    # - preparing the model
    logger.info(f"~> preparing the model (model type: {config['model']['type']})...\n")
    model = getattr(tabluence.deep_learning.pipeline.model, config['model']['type'])(
        config=config['model']['config'],
        tensorizer=tensorizer
    ).to(device)

    # - trainer
    logger.info(f"~> preparing the trainer (trainer type: {config['trainer']['type']})...\n")
    trainer = getattr(tabluence.deep_learning.pipeline.training, config['trainer']['type'])(
        config=config['trainer']['config'],
        dataloaders=dataloaders,
        model=model,
        device=device
    )

    # - starting the training procedure
    if not args.eval:
        logger.info(f"~> starting training sequence...\n")
        trainer.train()
    else:
        logger.info(f"~> starting evaluation...\n")
        trainer.eval()


if __name__ == "__main__":
    # - getting the arguments needed for training
    parser = get_train_parser()
    # - parsing the arguments
    args = parser.parse_args()

    # - running the training tool
    main(args=args)
