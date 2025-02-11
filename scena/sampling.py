from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from scena.model import Scena
from scena.modules.autoencoder import AutoEncoder
from scena.modules.conditioner import CLIPConditioner


@dataclass
class DenoisingInput:
    pass


def prepare(
    ae: AutoEncoder,
    clip: CLIPConditioner,
    imgs: torch.Tensor,
    input_masks: torch.Tensor,
    c2ws: torch.Tensor,
    Ks: torch.Tensor,
):
    pass


def denoise(model: Scena, input: DenoisingInput) -> torch.Tensor:
    pass


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, x.new_zeros([1])])


def make_beta_schedule(
    schedule,
    n_timestep,
    linear_start=1e-4,
    linear_end=2e-2,
    cosine_s=8e-3,
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule == "hf_linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timestep, dtype=torch.float64
        )
    else:
        raise NotImplementedError

    return betas.cpu().numpy()


class EpsScaling(object):
    def __call__(
        self, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = torch.ones_like(sigma, device=sigma.device)
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1.0) ** 0.5
        c_noise = sigma.clone()
        return c_skip, c_out, c_in, c_noise


def generate_roughly_equally_spaced_steps(
    num_substeps: int, max_step: int
) -> np.ndarray:
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False).astype(int)[::-1]


class LegacyDDPMDiscretization(object):
    def __init__(
        self,
        linear_start: Union[float, Tuple[float], str] = 0.00085,
        linear_end: Union[float, Tuple[float], str] = 0.0120,
        cosine_s: Union[float, Tuple[float], str] = 8e-3,
        num_timesteps: int = 1000,
        enforce_zero_terminal_snr: bool = False,
        schedule: str = "linear",
        log_snr_shift: Optional[Union[float, Tuple[float], str]] = None,
    ):
        linear_start = 0.00085
        linear_end = 0.012
        cosine_s = 0.008
        num_timesteps = 1000
        enforce_zero_terminal_snr = False
        schedule = "linear"
        # log_snr_shift = 2.4
        log_snr_shift = None
        self.num_timesteps = num_timesteps
        self.to_torch = partial(torch.tensor, dtype=torch.float32)

        if any(
            map(
                lambda x: isinstance(x, tuple),
                [linear_start, linear_end, cosine_s, log_snr_shift],
            )
        ):
            if not isinstance(linear_start, tuple):
                linear_start = (linear_start, linear_start)
            if not isinstance(linear_end, tuple):
                linear_end = (linear_end, linear_end)
            if not isinstance(cosine_s, tuple):
                cosine_s = (cosine_s, cosine_s)

            betas = []
            for s, e, c_s in zip(linear_start, linear_end, cosine_s):
                betas.append(
                    make_beta_schedule(
                        schedule,
                        num_timesteps,
                        linear_start=s,
                        linear_end=e,
                        cosine_s=c_s,
                    )
                )
            betas = np.stack(betas, axis=1)

            if not isinstance(log_snr_shift, tuple):
                log_snr_shift = (log_snr_shift, log_snr_shift)
            assert (
                log_snr_shift[0] >= log_snr_shift[1]
            ), "log_snr_shift should be in decreasing order if tuple"
            self.log_snr_shift = np.array([log_snr_shift])
        else:
            betas = make_beta_schedule(
                schedule,
                num_timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
            self.log_snr_shift = log_snr_shift

        if enforce_zero_terminal_snr:
            logpy.info("Enforcing zero terminal SNR")
            betas = enforce_zero_terminal_snr_converter(betas)
        alphas = 1.0 - betas  # first alpha here is on data side
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        if enforce_zero_terminal_snr:
            self.alphas_cumprod[-1] = 1.0e-10  # small eps

    def get_sigmas(self, n, device="cpu"):
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        if self.log_snr_shift is not None:
            sigmas = sigmas * np.exp(self.log_snr_shift)
        if sigmas.ndim == 2:
            linear_schedule = np.linspace(0, 1, n)
            sigmas = (
                linear_schedule * sigmas[:, 0] + (1 - linear_schedule) * sigmas[:, 1]
            )
        return torch.flip(
            to_torch(sigmas), (0,)
        )  # first sigma (idx 0) when indexed is on noise side

    def __call__(self, n, do_append_zero=True, device="cpu", flip=False):
        sigmas = self.get_sigmas(n, device=device)
        sigmas = append_zero(sigmas) if do_append_zero else sigmas
        return sigmas if not flip else torch.flip(sigmas, (0,))


class DiscreteDenoiser(nn.Module):
    def __init__(
        self,
        num_idx: int = 1000,
        do_append_zero: bool = False,
        quantize_c_noise: bool = True,
        flip: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.scaling = EpsScaling()
        self.discretization = LegacyDDPMDiscretization()
        self.num_idx = num_idx
        self.do_append_zero = do_append_zero
        self.flip = flip
        self.quantize_c_noise = quantize_c_noise
        self.device = device

        self.register_sigmas()

    def register_sigmas(self):
        sigmas = self.discretization(
            self.num_idx,
            do_append_zero=self.do_append_zero,
            flip=self.flip,
            device=self.device,
        )
        self.register_buffer("sigmas", sigmas)

    def sigma_to_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx: Union[torch.Tensor, int]) -> torch.Tensor:
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise

    def forward(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: dict,
        **additional_model_inputs,
    ) -> torch.Tensor:
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        if "replace" in cond:
            x, mask = cond.pop("replace").split((input.shape[1], 1), dim=1)
            input = input * (1 - mask) + x * mask
        return (
            network(input * c_in, c_noise, cond, **additional_model_inputs) * c_out
            + input * c_skip
        )


class VanillaCFG(object):
    def __init__(
        self,
        scale_rule_config: Optional[Union[Dict, List[Dict]]] = None,
        scale_schedule_config: Optional[Dict] = None,
        scale: Optional[float] = None,
        dyn_thresh_config: Optional[Dict] = None,
        additional_cond_keys: Optional[Union[List[str], str]] = None,
    ):
        scale_rule_config = default(
            scale_rule_config, {"target": "sgm.modules.diffusionmodules.guiders.NoRule"}
        )
        if isinstance(scale_rule_config, list):
            self.scale_rule = [
                instantiate_from_config(config) for config in scale_rule_config
            ]
        elif isinstance(scale_rule_config, dict):
            self.scale_rule = instantiate_from_config(scale_rule_config)
        else:
            raise ValueError(
                f"Invalid scale_rule_config type {type(scale_rule_config)}"
            )

        # assert (scale_schedule_config is None) != (scale is None)
        self.scale_schedule = instantiate_from_config(
            default(
                scale_schedule_config,
                {
                    "target": "sgm.modules.diffusionmodules.guiders.IdentitySchedule",
                    "params": {"scale": scale},
                },
            )
        )
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {
                    "target": "sgm.modules.diffusionmodules.guiders.NoDynamicThresholding"
                },
            )
        )

        additional_cond_keys = default(additional_cond_keys, [])
        if isinstance(additional_cond_keys, str):
            additional_cond_keys = [additional_cond_keys]
        self.additional_cond_keys = additional_cond_keys

    def __call__(
        self,
        x: torch.Tensor,
        sigma: Union[float, torch.Tensor],
        scale: Union[float, torch.Tensor] = None,
        return_unconditional: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        x_u, x_c = x.chunk(2)
        if isinstance(self.scale_rule, list):
            for rule in self.scale_rule:
                scale = rule(scale, **kwargs)
        else:
            scale = self.scale_rule(scale, **kwargs)
        scale_value = self.scale_schedule(sigma, scale, **kwargs)
        x_pred = self.dyn_thresh(x_u, x_c, scale_value, **kwargs)
        if return_unconditional:
            return x_pred, x_u
        return x_pred

    def prepare_inputs(
        self, x: torch.Tensor, s: torch.Tensor, c: Dict, uc: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat"] + self.additional_cond_keys:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out
