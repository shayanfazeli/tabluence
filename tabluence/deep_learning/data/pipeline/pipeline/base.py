from typing import List, Dict, Any, Iterator
import gzip, pickle
import os
import torch.utils.data.dataloader
import tabluence.deep_learning.data.pipeline.augmentation as augmentation_lib
import tabluence.deep_learning.data.pipeline.preprocessing as preprocessing_lib
import tabluence.deep_learning.data.pipeline.fusion as fusion_lib
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

libs = {
    'augmentation': augmentation_lib,
    'preprocessing': preprocessing_lib,
    'fusion': fusion_lib
}


class PipelinedIterableDataset(torch.utils.data.IterableDataset):
    """
    A dataset that is iterable and can be used with a DataLoader.

    Parameters
    ----------
    dataloader: `torch.utils.data.DataLoader`, required
        The dataloader that will be used.

    pipeline: `List[Any]`, required
        The list of modules (preprocessing, augmentation, or fusion instances) that will be applied to the data.
        The order matters.
    """
    def __init__(self, dataloader: torch.utils.data.dataloader.DataLoader, pipeline: List[Any]):
        """constructor"""
        self.dataloader = dataloader
        self.pipeline = pipeline

    def __iter__(self):
        for sample in self.dataloader:
            for module in self.pipeline:
                sample = module(sample)
            yield sample


class StandardDataSidePipeline:
    """
    The :cls:`StandardDataSidePipeline` is a base class for all pipelines on the dataside.
    """
    def __init__(
            self,
            module_configs: List[Dict[str, Any]],
            train_dataloader: torch.utils.data.dataloader.DataLoader,
            cache_filepath: str = None
    ):
        """Constructor"""
        self.module_configs = module_configs
        self.train_dataloader = train_dataloader
        self.modules = None
        # if not os.path.isfile(cache_filepath):
        self.build_modules()
        # self.save(filepath=cache_filepath)
        # else:
        #     logger.info(f"\t~> loading cache file from {cache_filepath}...\n")
        #     self.load(filepath=cache_filepath)

    def save(self, filepath: str) -> None:
        """
        save modules to a filepath
        """
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(self.modules, f)

    def load(self, filepath: str) -> None:
        """
        load modules from a filepath
        """
        with gzip.open(filepath, 'rb') as f:
            self.modules = pickle.load(f)

    def __call__(
            self,
            dataloader: torch.utils.data.dataloader.DataLoader,
            mode: str,
            modules: List[Any] = None,
    ) -> PipelinedIterableDataset:
        """
        The __call__ method of this class.

        Parameters
        ----------
        dataloader: `torch.utils.data.dataloader.DataLoader`, required
            The dataloader to be used for the data side operations.
        mode: `str`, required
            The mode of the pipeline, which is important as some of the modules are only applied in certain modes.
        modules: List[Any]
            The modules to be used for the data side operations.

        Returns
        -------
        `Iterator`: the new iterator formed by applying the modules on these.
        """
        if modules is None:
            assert mode in ['train', 'validation', 'test'], f"The mode (which is {mode}) must be one of 'train', 'validation', or 'test'."
            assert self.modules is not None, "Please run the `build_modules` method first to prepare, initialize, and learn the modules."
            modules = self.modules
            return PipelinedIterableDataset(
                dataloader=dataloader,
                pipeline=modules if mode == 'train' else [
                    modules[i] for i in range(len(modules)) if self.apply_in_eval_mode[i]
                ]
            )
        else:
            assert mode is None, "when modules are provided, mode must be set to None."
            return PipelinedIterableDataset(
                dataloader=dataloader,
                pipeline=modules
            )

    def build_modules(self) -> None:
        """
        The module that takes care of preparing the modules, setting them up, and learning them based on the
        given train dataloader.
        """
        self.apply_in_eval_mode = []
        self.modules = []
        # - building the modules from configs
        for config in self.module_configs:
            self.modules.append(
                getattr(libs[config['lib']], config['type'])(config=config['config'])
            )
            if config['lib'] == 'augmentation':
                self.apply_in_eval_mode.append(False)
            else:
                self.apply_in_eval_mode.append(True)

        for i in range(len(self.modules)):
            if self.module_configs[i]['lib'] in ['augmentation', 'preprocessing']:
                logger.info(f"""
                ~> learning the {i+1}/{len(self.modules)}th module from the
                dataside pipeline - an instance of `{self.module_configs[i]['lib']}` library of data side operations.
                """)
                self.modules[i].learn(
                    dataloader=self(
                        dataloader=self.train_dataloader,
                        modules=[self.modules[k] for k in range(i) if self.apply_in_eval_mode[k]],
                        mode=None
                    ),
                )
