from torch.utils.data import Sampler


DATASET_TO_LATENT_INDICES = {
    'dsprites': [1, 2, 3, 4, 5],
    'cars': [0, 1, 2],
    'dummy': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

DATASET_TO_FACTOR_SIZES = {
    'dsprites': [1, 3, 6, 40, 32, 32],
    'cars': [4, 24, 183],
    'dummy': [1] * 10
}