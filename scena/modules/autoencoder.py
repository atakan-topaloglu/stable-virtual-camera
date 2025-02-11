from typing import cast

import torch
from diffusers.models import AutoencoderKL
from torch import nn


class AutoEncoder(nn.Module):
    scale_factor: float = 0.18215
    downsample: int = 8

    def __init__(self):
        super().__init__()
        self.module = cast(
            AutoencoderKL,
            AutoencoderKL.from_pretrained(
                "stabilityai/stable-diffusion-2-1-base",
                subfolder="vae",
                force_download=False,
            ),
        )
        self.module.eval().requires_grad_(False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.module.encode(cast(torch.FloatTensor, x)).latent_dist.mean
            * self.scale_factor
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        __import__("ipdb").set_trace()
        return self.module.decode(z)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
