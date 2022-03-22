from typing import Optional, Tuple, Union, Generator
import numpy as np
from sklearn.utils.extmath import cartesian
from torch import Tensor
import torch
from torch.utils.data.dataset import Dataset
from dataset import DisentanglementDataset


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
        self.factor_bases = np.divide(
            self._num_feature_values, np.cumprod(self.factor_sizes)
        )
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
        data_source: DisentanglementDataset,
        device: torch.device,
        seed: Optional[int] = None,
    ) -> None:
        self.data_source = data_source
        self.device = device
        self.latent_indices = self.data_source.latent_indices
        self.factor_sizes = self.data_source.factor_sizes

        self.num_factors = len(self.factor_sizes)
        self.num_latents = len(self.latent_indices)

        self.observed_factor_indices = self._get_observed_indices()
        self.num_observed_factors = len(self.observed_factor_indices)

        self.features = self._get_features()
        self.feature_lookup = FeatureIndex(self.factor_sizes, self.features)

        self.seed = seed
        self.random_state = np.random.RandomState(seed)

    def _get_observed_indices(self):
        """Get all indices that are not given in the latent indices of the dataset (see DATASET_TO_LATENT_INDICES)."""
        indices = [i for i in range(self.num_factors) if i not in self.latent_indices]
        return indices

    def _get_features(self) -> np.ndarray:
        """Generates array containing all possible factor index combinations based on the size of each factor.
        Returned vector is of shape (np.prod(self.factor_sizes), num_factors).
        """
        return cartesian([np.array(list(range(i))) for i in self.factor_sizes])

    def sample_factors_of_variation(self, batch_size: int) -> np.ndarray:
        factors = np.zeros((batch_size, self.num_latents))
        for pos, idx in enumerate(self.latent_indices):
            # TODO: does pos work or should it be idx? (because idx should be the idx of the latent)
            factors[:, pos] = self._sample_factors(idx, batch_size)
        return factors

    def sample_all_factors(self, latent_factors: np.ndarray) -> np.ndarray:
        """Randomly samples any additional latent factor

        Only concats additional latent factors if
            `self.num_latents` < `self.num_factors`.
        This typically happens when a factor of variation only has one distinct
        value (like dSprites), therefore, there is no need to use the index of
        that latent factor to randomly sample from when computing metrics
        like beta-VAE.
        Parameters
        ----------
        latent_factors : np.ndarray
            A matrix of size (num_samples, self.num_latents) with randomly
            generated ground truth latent factors. This is typically from the
            output of `self.sample_latent_factors`.
        random_state : np.random.RandomState
            Pseudo-random number generator.
        Returns
        -------
        all_factors : np.ndarray
            Matrix of size (num_samples, self.num_factors) with randomly
            generated ground truth latent factors.
        """
        if self.num_observed_factors > 0:
            num_samples = len(latent_factors)
            all_factors = np.zeros((num_samples, self.num_factors))
            all_factors[:, self.latent_indices] = latent_factors
            for idx in self.observed_factor_indices:
                all_factors[:, idx] = self._sample_factors(idx, num_samples)
            return all_factors
        else:
            return latent_factors

    def sample_observations_from_factors(self, factors: np.ndarray) -> Tensor:
        """Randomly samples a batch of observations from a batch of factors."""
        all_factors = self.sample_all_factors(factors)
        indices = self.feature_lookup[all_factors]
        # TODO: return type?
        return self.data_source[indices].to(self.device)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, Tensor]:
        factors = self.sample_factors_of_variation(batch_size)
        observations = self.sample_observations_from_factors(factors)
        return factors, observations

    def generate(
        self, n_samples: int = 1000, batch_size: int = 64, drop_last: bool = False
    ) -> Generator[Tuple[np.ndarray, Tensor], None, None]:
        batches = [batch_size for _ in range(n_samples // batch_size)]
        if not drop_last and n_samples % batch_size != 0:
            batches.append(n_samples % batch_size)
        for batch in batches:
            yield self.sample(batch_size=batch)

    # TODO: change this to sample the range np.linspace()?
    def _sample_factors(self, idx, size):
        return self.random_state.randint(self.factor_sizes[idx], size=size)
