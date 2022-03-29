from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import torch
from evaluation.generator import LatentGenerator
from models import SoftIntroVAE
import ops


def generate_factor_representations(
    latent_generator: LatentGenerator,
    model: SoftIntroVAE,
    num_samples: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
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
    representations : np.ndarray
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
        with torch.no_grad():
            z, _ = model.encode(observations_batch)
            z = z.cpu().numpy()
        representations.append(z)

    return np.vstack(representations), np.vstack(factors)


# beta-vae
def generate_factor_change_batch(
    latent_generator: LatentGenerator, model: SoftIntroVAE, batch_size: int
) -> Tuple[np.ndarray, int]:
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
    v_li = latent_generator.sample_factors_of_variation(batch_size)
    v_lj = latent_generator.sample_factors_of_variation(batch_size)
    v_li[:, factor_index] = v_lj[:, factor_index]

    # Sim(vli, cli)
    x_li = latent_generator.sample_observations_from_factors(v_li)
    # Sim(vlj, clj)
    x_lj = latent_generator.sample_observations_from_factors(v_lj)

    # z_li = mu(x_li), z_lj = mu(x_lj)
    # real_mu, real_logvar, z, rec
    with torch.no_grad():
        z_mean_li, _ = model.encode(x_li)
        z_mean_lj, _ = model.encode(x_lj)

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


def compute_factor_change_accuracy(
    x_train, y_train, x_test, y_test, params=None
) -> float:
    """
    Calculates the factor change classification score proposed in
    https://openreview.net/references/pdf?id=Sy2fzU9gl.
    """
    params = params or {}
    lr_params = params.get("bvae_lr_params", {})
    if params.get("scale"):
        scl = StandardScaler()
        x_train = scl.fit_transform(x_train)
        x_test = scl.transform(x_test)

    clf = LogisticRegression(**lr_params)
    clf.fit(x_train, y_train)

    bvae_score: float = accuracy_score(y_test, clf.predict(x_test), normalize=True)
    return bvae_score


# DCI utils
def fit_info_clf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    params=None,
) -> Tuple[float, float, np.ndarray]:
    """
    Computes the informativeness score discussed on section 2 of
    https://openreview.net/pdf?id=By-7dz-AZ.
    """
    params = params or {}
    if params.get("informativeness_method") == "rf":
        estimator = RandomForestClassifier
    elif params.get("informativeness_method") == "xgb":
        estimator = XGBClassifier
    else:
        estimator = GradientBoostingClassifier
    estimator_params = params.get("informativeness_params", {})

    K = y_train.shape[1]
    feature_importances = []
    train_errors = []
    test_errors = []

    for i in range(K):
        y_train_i = y_train[:, i]
        y_test_i = y_test[:, i]

        clf = estimator(**estimator_params)
        clf.fit(x_train, y_train_i)

        train_errors.append(accuracy_score(y_train_i, clf.predict(x_train)))
        test_errors.append(accuracy_score(y_test_i, clf.predict(x_test)))
        feature_importances.append(np.abs(clf.feature_importances_))

    return np.mean(train_errors), np.mean(test_errors), np.array(feature_importances)


def compute_disentanglement(P: np.ndarray) -> float:
    """
    Computes the disentanlement score discussed on section 2 of
    https://openreview.net/pdf?id=By-7dz-AZ.
    """
    D = 1. - ops.entropy(P, base=P.shape[0])
    if np.sum(P) == 0:
        P = np.ones_like(P)
    ro = np.sum(P, axis=0) / P.sum()
    return np.sum(ro * D)


def compute_completeness(P: np.ndarray) -> float:
    """
    Computes the completeness score discussed on section 2 of
    https://openreview.net/pdf?id=By-7dz-AZ.
    """
    C = 1. - ops.entropy(P.T, base=P.shape[1])
    if np.sum(P) == 0:
        P = np.ones_like(P)
    ro = np.sum(P, axis=1) / P.sum()
    return np.sum(ro * C)