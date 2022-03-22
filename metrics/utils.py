from typing import Tuple
import numpy as np
import torch
from dataset import DisentanglementDataset
from metrics.generator import LatentGenerator
from models import SoftIntroVAE


DATASET_TO_LATENT_INDICES = {
    "dsprites": [1, 2, 3, 4, 5],
    "cars": [0, 1, 2],
    "dummy": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
}

DATASET_TO_FACTOR_SIZES = {
    "dsprites": [1, 3, 6, 40, 32, 32],
    "cars": [4, 24, 183],
    "dummy": [1] * 10,
}


def generate_factor_representations(
    latent_generator: LatentGenerator,
    model: SoftIntroVAE,
    num_samples: int,
    batch_size: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Randomly samples a set of observations (images) then returns
    their modeled latent representation and the ground truth latent factors
    they were generated from.
    Parameters
    ----------
    latent_generator : LatentGenerator
        Generator that has a `sample` method, which samples a
        batch of observations and the latent factors they were generated from.
    model : SoftIntroVAE
        Representation function that takes in an observation (image) and returns
        a latent representation.
    num_samples : int
        Number of latent representations to generate.
    batch_size : int
        Number of samples to generate per batch.
    Returns
    -------
    representations : torch.Tensor
        A (num_samples, latent_dim) size matrix where each row is the
        latent representation of a randomly sampled observation.
    factors : np.ndarray
        A (num_samples, latent_dim) size matrix where each row is the
        ground truth latent factors that generated the observation at the
        i^th index of `representations`.
    """
    representations = []
    factors = []
    for factors_batch, observations_batch in latent_generator.generate(
        num_samples, batch_size, drop_last=False
    ):
        factors.append(factors_batch)
        _, _, _, rec = model(observations_batch)
        representations.append(rec)

    return torch.vstack(representations), np.vstack(factors)


# beta-vae
def generate_factor_change_batch(
    latent_generator: LatentGenerator, model: SoftIntroVAE, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a single input (z_diff) and label (y) for a batch using the
    factor change algo described in A.4 of
    https://openreview.net/references/pdf?id=Sy2fzU9gl.
    Parameters
    ----------
    latent_generator : LatentGenerator
        Generator sample ground truth latent factors.
    model : SoftIntroVAE
        Encoder or representation function r(x) that takes in an input
        observation and returns the latent representation.
    batch_size : int
        Number of samples to generate per batch.
    Returns
    -------
    z_diff : np.ndarray
        Batch of z_diff scores calculated using the factor index
        labeled `y`.
    y : np.ndarray
        Index of the factor that remained unchanged when generating `z_diff`.
    """
    random_state = np.random.RandomState(latent_generator.seed)
    factor_index = random_state.randint(latent_generator.num_latents)  # y

    # sample ground truth factors and set a single factor v_k to
    # be the same (v_ik = v_jk)
    v_li = latent_generator.sample_factors_of_variation(batch_size, random_state)
    v_lj = latent_generator.sample_factors_of_variation(batch_size, random_state)
    v_li[:, factor_index] = v_lj[:, factor_index]

    # Sim(vli, cli)
    x_li = latent_generator.sample_observations_from_factors(v_li, random_state)
    # Sim(vlj, clj)
    x_lj = latent_generator.sample_observations_from_factors(v_lj, random_state)

    # z_li = mu(x_li), z_lj = mu(x_lj)
    # real_mu, real_logvar, z, rec
    z_mean_li, _, _, _ = model(x_li)
    z_mean_lj, _, _, _ = model(x_lj)

    z_mean_li: np.ndarray = z_mean_li.cpu().numpy()
    z_mean_lj: np.ndarray = z_mean_lj.cpu().numpy()

    # z_diff = 1/L * sum_l (|z_li - z_lj|)
    z_mean_li = z_mean_li.reshape(batch_size, -1)
    z_mean_lj = z_mean_lj.reshape(batch_size, -1)
    z_diff = np.mean(np.abs(z_mean_li - z_mean_lj), axis=0)

    return z_diff, factor_index


def generate_factor_change(
    latent_generator: LatentGenerator,
    model: SoftIntroVAE,
    num_samples: int,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates `num_batches` inputs (z_diff) and labels (y) for using the
    factor change algo proposed in A.4 of
    https://openreview.net/references/pdf?id=Sy2fzU9gl.
    Parameters
    ----------
    latent_generator : LatentGenerator
        Generator sample ground truth latent factors.
    model : SoftIntroVAE
        Encoder or representation function r(x) that takes in an input
        observation and returns the latent representation.
    num_samples : int
        Number of latent representations to generate.
    batch_size : int (default=64)
        Number of samples to generate per batch.
    random_state : np.random.RandomState
        Pseudo-random number generator.
    Returns
    -------
    z_diff : np.ndarray
        Matrix of shape (`num_batches`, 1) where each row is the z_diff score
        calculated using the factor index labeled `y`.
    y : np.ndarray
        Index of the factor that remained unchanged when generating z_diff.
    """
    Z_diff = []
    y = []
    num_batches = int(np.ceil(num_samples / batch_size))

    for _ in range(num_batches):
        z_diff_batch, y_batch = generate_factor_change_batch(
            latent_generator, model, batch_size=batch_size
        )
        Z_diff.append(z_diff_batch)
        y.append(y_batch)

    return np.array(Z_diff, dtype=np.float32), np.array(y, dtype=np.int8)
