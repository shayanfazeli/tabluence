from typing import Dict, Any, List
import abc
import torch
import torch.nn
import torch.optim
import torch.utils.data.dataloader
import logging

from tabluence.deep_learning.data.pipeline.pipeline import StandardDataSidePipeline

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TrainerBase(metaclass=abc.ABCMeta):
    """
    Base class for all trainers.
    """
    def __init__(
            self,
            device: torch.device,
            config: Dict[str, Any],
            model: torch.nn.Module,
            dataloaders: Dict[str, torch.utils.data.dataloader.DataLoader]
    ):
        self.dataloaders = dataloaders
        self.config = config
        self.model = model
        self.start_epoch = 0
        self.device = device
        self.building_blocks()
        self.buffering_initialization()

    @abc.abstractmethod
    def buffering_initialization(self):
        """
        Initializing the buffering mechanism.
        """
        pass

    @abc.abstractmethod
    def iteration_buffering(self):
        """
        Buffering the data for the current iteration.
        """
        pass

    @abc.abstractmethod
    def buffering_reset(self):
        """
        Resetting the buffering mechanism.
        """
        pass

    @abc.abstractmethod
    def process_buffers(self):
        pass

    @abc.abstractmethod
    def train_epoch(self, epoch_index: int) -> Dict[str, Any]:
        """
        training mechanism for the entire epoch.
        """
        pass

    @abc.abstractmethod
    def evaluate_epoch(self, epoch_index: int) -> Dict[str, Any]:
        """
        epoch evaluation
        """
        pass

    @abc.abstractmethod
    def building_blocks(self) -> None:
        """
        Building the blocks of the trainer model.
        """
        pass

    @abc.abstractmethod
    def checkpointing_dump(self, epoch_index: int):
        """
        checkpointing dump
        """
        pass

    @abc.abstractmethod
    def checkpointing_resume(self):
        pass

    def train(self):
        self.checkpointing_resume()
        for epoch_index in range(self.start_epoch, self.config['max_epochs']):
            self.buffering_reset()
            logger.info(f"EPOCH {epoch_index} - TRAINING\n\n")
            self.train_epoch(epoch_index)

            logger.info(f"EPOCH {epoch_index} - TESTING\n\n")
            self.evaluate_epoch(epoch_index)

    def eval(self):
        self.checkpointing_resume()
        epoch_index = self.start_epoch - 1
        logger.info(f"Evaluating the model that was trained for {epoch_index} epochs")
        self.evaluate_epoch(epoch_index)
