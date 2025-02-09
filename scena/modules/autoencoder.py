from diffusers.models import AutoencoderKL
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        module = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="vae"
        )
