from sklearn.preprocessing import MinMaxScaler
import logging
from tabluence.deep_learning.data.pipeline.preprocessing.single_slice.normalization.base import DataSideNormalizationBase

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MinMaxSingleSliceNormalization(DataSideNormalizationBase):
    """
    :cls:`MinMaxSingleSliceNormalization` is the class for min-max normalization of single slice data.

    In this version, a single configuration is used for all features across all slices.
    Please modify accordingly if needed.
    """
    def __init__(self, **kwargs):
        """
        Constructor
        """
        super(MinMaxSingleSliceNormalization, self).__init__(**kwargs)

    @property
    def normalizer_model(self):
        return MinMaxScaler

    @property
    def normalizer_args(self):
        return dict()
