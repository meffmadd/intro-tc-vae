from evaluation.generator import LatentGenerator
from models import SoftIntroVAE
from . import utils


def compute_bvae_score(
    latent_generator: LatentGenerator,
    model: SoftIntroVAE,
    num_samples: int = 10000,
    batch_size: int = 64,
    params=None,
):
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
    params : dict (default=None)
        bvae_lr_params :
        scale :
    Returns
    -------
    bvae_score : float
        beta-VAE disentanglement score.
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
        Z_diff_train, y_train, Z_diff_test, y_test, params=params
    )
    return bvae_score
