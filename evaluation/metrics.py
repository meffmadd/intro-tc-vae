from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
from evaluation.generator import LatentGenerator
from models import SoftIntroVAE
from . import utils
from torch.utils.tensorboard import SummaryWriter


def write_bvae_score(writer: SummaryWriter, cur_iter: int, **score_kwargs):
    bvae_score, bvae_score_scaled = compute_bvae_score(**score_kwargs)
    writer.add_scalars(
        "bvae_score",
        dict(score=bvae_score, scaled=bvae_score_scaled),
        global_step=cur_iter,
    )


def compute_bvae_score(
    latent_generator: LatentGenerator,
    model: SoftIntroVAE,
    num_samples: int = 10000,
    batch_size: int = 64,
) -> Tuple[float, float]:
    """beta-VAE Disentanglement Metric
    Section 3 of "beta-VAE: Learning Basic Visual Concepts with a Constrained
    Variational Framework" (https://openreview.net/references/pdf?id=Sy2fzU9gl).
    Factor change classification
    ----------------------------
    Let L denote the number of latent variable,
        (1) Randomly sample ground truth factors v_li and v_lj such that a
            single factor v_k remains unchanged for both.
        (2) Generate observations x_li and x_lj from v_li and v_lj.
        (3) Generate z_li = mu(x_li) and z_lj = mu(x_lj), where mu(x) returns
            the mean latent vector from the representation function (encoder).
        (4) Calculate z_diff as,
                z_diff = 1/L * sum_l (|z_li - z_lj|).
        (5) bvae_score = ACC(y, p(y|z_diff))
                where p is a linear classifier trained to predict which
                factor y remained unchanged when generating z_diff.
    Parameters
    ----------
    latent_generator : LatentGenerator
        Generator sample ground truth latent factors.
    model : SoftIntroVAE
        Encoder or representation function r(x) that takes in an input
        observation and returns the latent representation.
    num_samples : (default=10000)
        Number of latent representations to generate.
    batch_size : int (default=64)
        Number of samples to generate per batch.
    Returns
    -------
    bvae_score : float
        beta-VAE disentanglement score.
    bvae_score_scaled : float
        beta-VAE disentanglement score with scaling of data.
    """
    Z_diff_train, y_train = utils.generate_factor_change(
        latent_generator,
        model,
        num_samples,
        batch_size=batch_size,
    )
    Z_diff_test, y_test = utils.generate_factor_change(
        latent_generator,
        model,
        num_samples,
        batch_size=batch_size,
    )

    bvae_score = utils.compute_factor_change_accuracy(
        Z_diff_train, y_train, Z_diff_test, y_test, params=dict(scale=False)
    )
    bvae_score_scaled = utils.compute_factor_change_accuracy(
        Z_diff_train, y_train, Z_diff_test, y_test, params=dict(scale=True)
    )
    return bvae_score, bvae_score_scaled


def write_dci_score(writer: SummaryWriter, cur_iter: int, **score_kwargs):
    dci_info_score, dci_comp_score, dci_dis_score = compute_dci_score(
        **score_kwargs,
        params=dict(
            informativeness_method="xgb",
            informativeness_params=dict(
                tree_method="gpu_hist",
                gpu_id=0,
                eval_metric="mlogloss",
                use_label_encoder=False,
            ),
        ),
    )
    writer.add_scalars(
        "dci",
        dict(
            dci_informativeness_score=dci_info_score,
            dci_completeness_score=dci_comp_score,
            dci_disentanglement_score=dci_dis_score,
        ),
        global_step=cur_iter,
    )


def compute_dci_score(
    latent_generator: LatentGenerator,
    model: SoftIntroVAE,
    num_samples=10000,
    batch_size=64,
    params=None,
) -> Tuple[float, float, float]:
    """Disentanglement, Completeness, and Informativeness (DCI)
    Section 2 of "A Framework for the Quantitative Evaluation of Disentangled
    Representations" (https://openreview.net/pdf?id=By-7dz-AZ).
    Let P be a (D, K) matrix where the each ij^th element denotes the
    probability of latent variable c_i of dimension D being important for
    predicting latent factor z_j of dim K. Let ro be a vector of length K
    with a weighted average from each factor.
    D = sum_i (ro_i * (1 - H(P_i)))
    C = sum_j (ro_j * (1 - H(P_j)))
    I = E(z_j, f_j(c)), i.e. the prediction error to predict z_j from c.
    Parameters
    ----------
    latent_generator : LatentGenerator
        Generator sample ground truth latent factors.
    model : SoftIntroVAE
        Encoder or representation function r(x) that takes in an input
        observation and returns the latent representation.
    num_samples : (default=10000)
        Number of latent representations to generate.
    batch_size : int (default=64)
        Number of samples to generate per batch.
    Returns
    -------
    dci_info_score : float
        Informativeness score.
    dci_comp_score : float
        Completeness score.
    dci_dis_score : float
        Disentanglement score.
    """
    params = params or {}

    x_train, y_train = utils.generate_factor_representations(
        latent_generator,
        model,
        num_samples=num_samples,
        batch_size=batch_size,
    )
    x_test, y_test = utils.generate_factor_representations(
        latent_generator,
        model,
        num_samples=num_samples,
        batch_size=batch_size,
    )
    _, test_error, P = utils.fit_info_clf(
        x_train, y_train, x_test, y_test, params=params
    )

    return test_error, utils.compute_completeness(P), utils.compute_disentanglement(P)


def write_mig_score(writer: SummaryWriter, cur_iter: int, **score_kwargs):
    mig_score = compute_mig_score(**score_kwargs)
    writer.add_scalar("mig_score", mig_score, global_step=cur_iter)


def compute_mig_score(
    latent_generator: LatentGenerator,
    model: SoftIntroVAE,
    num_samples=10000,
    batch_size=64,
    params=None,
):
    """Mutual Information Gap (MIG)
    Equation 6 (section 4.1) of "Isolating Sources of Disentanglement in
    Variational Autoencoders" (https://arxiv.org/pdf/1802.04942.pdf).
    Let z_j denote the jth latent variable, v_j the jth ground truth latent
    factor, and K the number of ground truth factors.
    MIG = 1/K * sum_k (1 / H(v_k)) * (argmax_jk( I(z_jk ; v_k) ) - argmax_j( I(z_j ; v_k)) )
    Where argmax_jk( I(z_jk ; v_k) ) denotes is the highest mutual
    information (MI) between the jk^th latent and k^th factor.
    argmax_j( I(z_j ; v_k) ) denotes the 2nd highest MI between the jth latent
    and kth factor.
    Parameters
    ----------
    latent_generator : LatentGenerator
        Generator sample ground truth latent factors.
    model : SoftIntroVAE
        Encoder or representation function r(x) that takes in an input
        observation and returns the latent representation.
    num_samples : (default=10000)
        Number of latent representations to generate.
    batch_size : int (default=64)
        Number of samples to generate per batch.
    params : dict (default=None)
        bins : Discrete number of bins to encode each latent variable.
    Returns
    -------
    mig_score : float
        Mutual Information Gap score.
    """
    params = params or {}
    bins = params.get("bins", 10)

    mu, v = utils.generate_factor_representations(
        latent_generator,
        model,
        num_samples=num_samples,
        batch_size=batch_size,
    )
    mu_binned = utils.discretize(mu, bins=bins)

    H = utils.calculate_entropy(v)
    I = utils.calculate_mutual_info(mu_binned, v)
    I_sorted = np.sort(I, axis=0)[::-1]

    return np.mean((I_sorted[0] - I_sorted[1]) / H)


def write_mod_expl_score(writer: SummaryWriter, cur_iter: int, **score_kwargs):
    modularity_score, explicitness_score = compute_mod_expl_score(
        **score_kwargs,
        params=dict(explicitness_lr_params={"solver": "saga", "max_iter": 300}),
    )
    writer.add_scalars(
        "mod_expl",
        dict(
            modularity_score=modularity_score,
            explicitness_score=explicitness_score,
        ),
        global_step=cur_iter,
    )


def compute_mod_expl_score(
    latent_generator: LatentGenerator,
    model: SoftIntroVAE,
    num_samples=10000,
    batch_size=64,
    params=None,
):
    """Modularity and Explicitness scores.
    Equation 2 (Section 3) of "Learning Deep Disentangled Embeddings with
    the F-Statistic Loss" (https://arxiv.org/pdf/1802.05312.pdf).
    Let m_if denote the mutual information (MI) between the latent i and
    factor j. Then t is a zero matrix of the same size called a "template"
    where each row has only one non-zero element, the highest m_ij for latent i.
    Let N denotes the number of factors
    modularity = sum_i (1 - delta_i) / N
        s.t. delta_i = sum_f ((m_if - t_if)^2) / (theta^2_i * (N - 1))
    explicitness = sum_j (AUC(z_j, f_j(v))) / N
        where j is a factor index of z and k is an index on values of factor j.
        i.e. each factor is one hot encoded and we're trying to predict
        with each latent, then taking the mean AUC to evaluate performance.
    Parameters
    ----------
    latent_generator : LatentGenerator
        Generator sample ground truth latent factors.
    model : SoftIntroVAE
        Encoder or representation function r(x) that takes in an input
        observation and returns the latent representation.
    num_samples : (default=10000)
        Number of latent representations to generate.
    batch_size : int (default=64)
        Number of samples to generate per batch.
    params : dict (default=None)
        bins : Discrete number of bins to encode each latent variable.
    Returns
    -------
    modularity_score : float
        Modularity score.
    explicitness_score : float
        Explicitness score.
    """
    params = params or {}
    bins = params.get("bins", 20)

    x_train, y_train = utils.generate_factor_representations(
        latent_generator,
        model,
        num_samples=num_samples,
        batch_size=batch_size,
    )
    x_test, y_test = utils.generate_factor_representations(
        latent_generator,
        model,
        num_samples=num_samples,
        batch_size=batch_size,
    )

    x_train_binned = utils.discretize(x_train, bins=bins)
    MI = utils.calculate_mutual_info(x_train_binned, y_train)

    scl = StandardScaler()
    x_train = scl.fit_transform(x_train)
    x_test = scl.transform(x_test)

    _, val_explicit_score = utils.compute_explicitness(
        x_train, y_train, x_test, y_test, params=params
    )

    return utils.compute_modularity(MI), val_explicit_score
