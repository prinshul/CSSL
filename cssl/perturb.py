import math
from typing import List, Optional

import torch


class AdaptivePerturbation:
    def __init__(self, epsilon: float = 0.05, momentum: float = 0.99, relative: bool = True,
                 init_omega: float = 0.0):
        self.epsilon = epsilon
        self.momentum = momentum
        self.relative = relative
        self.omega = init_omega

    @property
    def lam(self) -> float:
        return (self.omega + 1.0) / 2.0 + self.epsilon

    def update(self, omega: Optional[float]) -> None:
        if omega is None or not math.isfinite(omega):
            return
        self.omega = self.momentum * self.omega + (1.0 - self.momentum) * float(omega)

    def __call__(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        std = math.sqrt(self.lam)
        out = []
        for f in feats:
            noise = torch.randn_like(f) * std
            if self.relative:
                noise = noise * f.detach().float().std(dim=(1, 2, 3), keepdim=True).to(f.dtype)
            out.append(f + noise)
        return out

    def state_dict(self):
        return {"omega": self.omega}

    def load_state_dict(self, d):
        self.omega = float(d["omega"])
