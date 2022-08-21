from typing import Dict, Any
from sklearn.mixture import GaussianMixture
import logging
from tabluence.deep_learning.data.pipeline.augmentation.single_slice.sample_from_distribution.base import \
    SampleFromDistributionAugmentionBase

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GaussianMixturesSingleSliceAugmentation(SampleFromDistributionAugmentionBase):
    """
    :cls:`GaussianMixturesSingleSliceAugmentation` is the class for augmentation
    and imputation of single slice data using Gaussian Mixture Model.

    In this version, a single configuration is used for all features across all slices.
    Please modify accordingly if needed.

    Please note that the configuration for this module needs to contain the following keys:
        - gmm : `Dict[str, Any]`, required
            The parameters for the :cls:`sklearn.mixture.GaussianMixture` model.
    """
    def __init__(self, *args, **kwargs):
        """
        Constructor
        """
        super(GaussianMixturesSingleSliceAugmentation, self).__init__(*args, **kwargs)
        self.gmm_config = self.config['gmm']

    @property
    def distribution_model(self):
        return GaussianMixture

    @property
    def distribution_args(self):
        return self.gmm_config
