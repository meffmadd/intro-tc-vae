from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
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

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

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
    D = 1.0 - ops.entropy(P, base=P.shape[0])
    if np.sum(P) == 0:
        P = np.ones_like(P)
    ro = np.sum(P, axis=0) / P.sum()
    return np.sum(ro * D)


def compute_completeness(P: np.ndarray) -> float:
    """
    Computes the completeness score discussed on section 2 of
    https://openreview.net/pdf?id=By-7dz-AZ.
    """
    C = 1.0 - ops.entropy(P.T, base=P.shape[1])
    if np.sum(P) == 0:
        P = np.ones_like(P)
    ro = np.sum(P, axis=1) / P.sum()
    return np.sum(ro * C)


# MIG utils
def discretize(x, bins):
    """Discretizes each column of x using a histogram function."""
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    descretized = np.zeros(x.shape)
    for i in range(x.shape[1]):
        _, bin_edges = np.histogram(x[:, i], bins)
        descretized[:, i] = np.digitize(x[:, i], bin_edges[:-1])
    return descretized


def calculate_mutual_info(z, v):
    """Calcultes the mutual information between each latent z and factor v."""
    n = z.shape[1]
    d = v.shape[1]
    MI = np.zeros([n, d])
    for i in range(n):
        for j in range(d):
            MI[i, j] = mutual_info_score(z[:, i], v[:, j])
    return MI


def calculate_entropy(v):
    """Calculates the entropy of each column (factor) of a matrix v."""
    d = v.shape[1]
    H = np.zeros(d)
    for j in range(d):
        H[j] = mutual_info_score(v[:, j], v[:, j])
    return H


# Modular & Explicitness utils
def get_valid_indices(y_train, y_test):
    """Mask values of y_train and y_test that are not contained in both y_train and y_test."""
    labels = np.array(list(set(y_train) & set(y_test)))
    train_idx = list(map(lambda x: x in labels, y_train))
    test_idx = list(map(lambda x: x in labels, y_test))
    return train_idx, test_idx


def compute_explicitness(x_train, y_train, x_test, y_test, params=None):
    """
    Computes the explicitness shown in section 3
    of https://arxiv.org/pdf/1802.05312.pdf.
    """
    params = params or {}
    lr_params = params.get("explicitness_lr_params", {})

    num_factors = y_train.shape[1]

    train_aucs = []
    test_aucs = []
    # may want to test splitting ever level into a binary logistic
    for i in range(num_factors):
        y_train_i = y_train[:, i].astype(int)
        y_test_i = y_test[:, i].astype(int)

        train_idx, test_idx = get_valid_indices(y_train_i, y_test_i)
        x_train_i, y_train_i = x_train[train_idx, :], y_train_i[train_idx]
        x_test_i, y_test_i = x_test[test_idx, :], y_test_i[test_idx]

        clf = LogisticRegression(**lr_params)
        clf.fit(x_train_i, y_train_i)

        y_pred = clf.predict_proba(x_train_i)
        y_pred_test = clf.predict_proba(x_test_i)

        # one-hot encoding to calculate AUC with output of predict_proba (prob. for each class)
        mlb = MultiLabelBinarizer()
        y_train_enc = mlb.fit_transform(y_train_i.reshape(-1, 1))
        y_test_enc = mlb.transform(y_test_i.reshape(-1, 1))

        train_aucs.append(roc_auc_score(y_train_enc, y_pred))
        test_aucs.append(roc_auc_score(y_test_enc, y_pred_test))

    return np.mean(train_aucs), np.mean(test_aucs)


def compute_modularity(mi):
    """
    Computes the modularity score given a mutual information matrix shown in
    section 3 using equation 2 of https://arxiv.org/pdf/1802.05312.pdf.
    """
    num_latents = mi.shape[0]
    N = mi.shape[1]
    template = np.zeros_like(mi)
    max_mi_idx = np.argmax(mi, axis=1)
    thetas = np.max(mi, axis=1)
    template[range(num_latents), max_mi_idx] = thetas
    deltas = np.sum((mi - template) ** 2, axis=1) / (thetas**2 * (N - 1))
    return np.mean(1 - deltas)
