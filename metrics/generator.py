from typing import Sized, Union
import numpy as np
from sklearn.utils.extmath import cartesian
from torch import Tensor
from torch.utils.data.dataset import Dataset
from models import SoftIntroVAE


class FeatureIndex(object):
    """Serves as a lookup dictionary that returns the index of an image given
    a factor configuration.

    This only works if the dataset is structured based on the factors. 
    The factors serve as counting, starting from the "least significant" factor to the "most significant" factor.
    For example, if the data contains only factors with 2 possible values the data is counted like binary and 
    the resulting index is the representation of the factors in binary.

    The factor_bases represent how far to "jump" in the index if the value is given. 
    From the previous example with binary factors (lets say 5), if the first factor is 1 then we know the
    index has to be greater than 16 and similar with the other factors (that's what the dot product does).

    """
    def __init__(self, factor_sizes: Union[list, np.ndarray], features=None):
        self.factor_sizes = factor_sizes
        self.features = features
        self._num_feature_values = np.prod(self.factor_sizes)
        self.factor_bases = np.divide(self._num_feature_values,
                                      np.cumprod(self.factor_sizes))
        self._features_to_index = np.arange(self._num_feature_values)

    def _get_feature_space(self, features):
        return np.dot(features, self.factor_bases).astype(np.int32)

    def __len__(self):
        return len(self._features_to_index)

    def __getitem__(self, features):
        """
        Given a batch of ground truth latent factors returns the indices
        of the images they generate.
        """
        return self._features_to_index[self._get_feature_space(features)]

    def keys(self):
        return self._features_to_index

    def values(self):
        return self.features

    def items(self):
        return zip(self.keys(), self.values())


class LatentGenerator:
    def __init__(
        self,
        model: SoftIntroVAE,
        data_source: Dataset,
        latent_sizes: Union[list, np.ndarray],
        factor_sizes: Union[list, np.ndarray],
    ) -> None:
        self.model = model
        self.data_source = data_source
        self.latent_indices = latent_sizes
        self.factor_sizes = factor_sizes

        self.num_factors = len(self.factor_sizes)
        self.num_latents = len(self.latent_indices)

    def _get_observed_indices(self):
        indices = [
            i
            for i
            in range(self.num_factors)
            if i not in self.latent_indices
        ]
        return indices

    def _get_features(self) -> np.ndarray:
        """ Generates array containing all possible factor index combinations based on the size of each factor.
        Returned vector is of shape (np.prod(self.factor_sizes), num_factors).
        """
        return cartesian([np.array(list(range(i)))
                          for i in
                          self.factor_sizes])

    def generate_factor_change_batch(batch_size: int) -> Tensor:
        pass

    def sample_observations_from_factors(factors: Tensor) -> Tensor:
        pass
