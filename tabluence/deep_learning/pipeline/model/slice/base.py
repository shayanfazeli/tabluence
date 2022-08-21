from typing import Dict, Any, List
import pandas
import abc
import torch

from tabluence.deep_learning.data.tensorizer.base import TensorizerBase


class SliceModelBase(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    The base class for the model instances designed to work with the single slice
    dataset format.

    The steps to create these model instances are as follows:

    * Create a subclass of this class.
    * Note that the `config` object will have to be passed to the constructor.
    * Override the :meth:`building_blocks` method, in which all of the modules
    required for the model will be made.
    * Override the :meth:`initialize_weights` method for any customized initialization.
    * Override the :meth:`preprocess_batch` for any customized preprocessing. Please note that
    if tensorization is required, its place would be in the :meth:`preprocess_batch` method.
    * Override the :meth:`inference_train` method for the application of the model on the processed
    batch data.
    * Override the :meth:`inferencce_eval` method for evaluation. Please note that chances are this mechanism
    be exactly identical too the `inference_train`, in that case, just run that under the `torch.no_grad()`.
    * Override the :meth:`loss` method for your model, and note that it shall work with the outputs of
    :meth:`inference_train` method. Please note that its output is usually expected to have the key `loss`,
    accessible via `outputs['loss_outputs']['loss']` when `outputs = model.forward_train(...)`.
    """
    def __init__(self, config: Dict[str, Any], tensorizer: TensorizerBase):
        super(SliceModelBase, self).__init__()
        self.config = config
        self.building_blocks()
        self.initialize_weights()
        self.add_module('tensorizer', tensorizer)

    @abc.abstractmethod
    def initialize_weights(self) -> None:
        """
        For initializing the weights
        """
        pass

    @abc.abstractmethod
    def preprocess_batch(self, batch: Dict[str, List[Dict[str, pandas.DataFrame]]]) -> Dict[str, Any]:
        """
        This method will be used

        Parameters
        ----------
        batch: `Dict[str, List[Dict[str, pandas.DataFrame]]]`, required
            The batch coming form a single slice dataset.

        Returns
        -------
        `Dict[str, Any]`: the processed bundle, which can be different per model.
        """
        pass

    @abc.abstractmethod
    def loss(self, batch: Any, model_outputs: Any) -> Dict[str, torch.Tensor]:
        """
        This interface is to be implemented by the child class.
        It must contain the routine for calculating the loss.

        Parameters
        ----------
        batch: `Dict[str, List[Dict[str, pandas.DataFrame]]]`, required
            The batch coming form a single slice dataset.

        model_outputs: `Dict[str, torch.Tensor]`, required
            The model outputs which is in the form of an information bundle
            dictionary.

        Returns
        -------
        `Dict[str, torch.Tensor]`: the information bundle of the forward process.
        """
        pass

    @abc.abstractmethod
    def inference_train(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        The inference making with preservation of gradients, for training.

        Parameters
        ----------
        batch: `Dict[str, List[Dict[str, pandas.DataFrame]]]`, required
            The batch coming form a single slice dataset.

        Returns
        -------
        `Dict[str, torch.Tensor]`: the information bundle of the forward process.
        """
        pass

    @abc.abstractmethod
    def inference_eval(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        The inference making (please note it is caller's responsibility to set the model to eval mode)

        Parameters
        ----------
        batch: `Dict[str, List[Dict[str, pandas.DataFrame]]]`, required
            The batch coming form a single slice dataset.

        Returns
        -------
        `Dict[str, torch.Tensor]`: the information bundle of the forward process.
        """
        pass

    @abc.abstractmethod
    def building_blocks(self) -> None:
        """
        This interface is to be implemented by the child class.
        It must contain the routine for adding the necessary building blocks (layers, etc.)
        """
        pass

    def forward_train(self, batch: Dict[str, List[Dict[str, pandas.DataFrame]]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        batch: `Dict[str, List[Dict[str, pandas.DataFrame]]]`, required
            The batch coming form a single slice dataset.

        Returns
        -------
        `Dict[str, Dict[str, torch.Tensor]]`: the information bundle of the forward process.
        """
        self.train()
        # - preprocessing
        batch = self.preprocess_batch(batch)

        # - passing through inference
        model_outputs = self.inference_train(batch)

        # - computing losses
        loss_outputs = self.loss(batch, model_outputs)

        # - returning the information bundle
        return dict(
            model_outputs=model_outputs,
            loss_outputs=loss_outputs
        )

    def forward_eval(self, batch: Dict[str, List[Dict[str, pandas.DataFrame]]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        The evaluation forward process

        Parameters
        ----------
        batch: `Dict[str, List[Dict[str, pandas.DataFrame]]]`, required
            The batch coming form a single slice dataset.

        Returns
        -------
        `Dict[str, Dict[str, torch.Tensor]]`: the information bundle of the forward process.
        """
        batch = self.preprocess_batch(batch)
        self.eval()

        with torch.no_grad():
            # - passing through inference
            model_outputs = self.inference_eval(batch)

            # - computing losses
            loss_outputs = self.loss(batch, model_outputs)

            # - returning the information bundle
            return dict(
                model_outputs=model_outputs,
                loss_outputs=loss_outputs
            )

    def forward(self, batch: Dict[str, List[Dict[str, pandas.DataFrame]]], mode: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        batch: `Dict[str, List[Dict[str, pandas.DataFrame]]]`, required
            The batch coming form a single slice dataset.
        mode: `str`, required
            The mode of the forward process.

        Returns
        -------
        `Dict[str, Dict[str, torch.Tensor]]`: the information bundle of the forward process.
        """
        if mode == 'train':
            return self.forward_train(batch)
        elif mode in ['test', 'eval', 'validation', 'val']:
            return self.forward_eval(batch)
        else:
            raise ValueError(f'Unknown mode: {mode}')
