from torch import Tensor
from solvers.intro import IntroSolver
from solvers.tc import TCSovler


class IntroTCSovler(IntroSolver):
    def compute_kl_loss(self, z: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        return TCSovler.compute_kl_loss(self, z, mu, logvar)
