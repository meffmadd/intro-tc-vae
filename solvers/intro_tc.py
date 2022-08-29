from typing import Optional
from torch import Tensor
from solvers.intro import IntroSolver
from solvers.tc import TCSovler


class IntroTCSovler(IntroSolver):
    def compute_kl_loss(
        self,
        z: Optional[Tensor],
        mu: Tensor,
        logvar: Tensor,
        reduce: str = "mean",
        beta: float = None,
    ) -> Tensor:
        return TCSovler.compute_kl_loss(self, z, mu, logvar, reduce, beta)
