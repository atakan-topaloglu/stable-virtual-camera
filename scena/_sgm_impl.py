import functools
import importlib
import logging
import math
from abc import abstractmethod
from copy import deepcopy
from functools import partial
from inspect import isfunction
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import OmegaConf
from packaging import version
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    ActivationWrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.utils.checkpoint import checkpoint

T = TypeVar("T")

logpy = logging.getLogger(__name__)

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}
    logpy.warn(
        f"No SDP backend available, likely because you are running in pytorch "
        f"versions < 2.0. In fact, you are using PyTorch {torch.__version__}. "
        f"You might want to consider upgrading."
    )

import xformers
import xformers.ops

XFORMERS_IS_AVAILABLE = True


def exists(x: Any) -> bool:
    return x is not None


def default(val: Optional[T], d: Union[Callable[[], T], T]) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d


def get_obj_from_str(
    string: str, reload: bool = False, invalidate_cache: bool = True
) -> Any:
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: dict) -> Any:
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    if OmegaConf.is_config(config):
        logpy.warn(
            f"You called instantiate_from_config with OmegaConf wrapped dictionary "
            f"that may introduce problems in instantiated class. Converting to "
            f"python dict instead. Was called for class {config['target']}."
        )
        config = OmegaConf.to_container(config, resolve=True)
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Timestep(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return timestep_embedding(t, self.dim)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        dense_emb: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        time_context: Optional[int] = None,
        num_video_frames: Optional[int] = None,
    ):
        for layer in self:
            if isinstance(layer, ActivationWrapper):
                module = layer._checkpoint_wrapped_module
            elif isinstance(layer, FullyShardedDataParallel):
                module = layer.module
            else:
                module = layer

            if isinstance(module, TimestepBlock) and not isinstance(
                module, PostHocResBlockWithTime
            ):
                x = layer(x, emb, dense_emb)
            elif isinstance(module, PostHocResBlockWithTime):
                x = layer(x, emb, dense_emb, num_video_frames, image_only_indicator)
            elif isinstance(
                module,
                (
                    PostHocSpatialTransformerWithTimeMixing,
                    PostHocAttentionBlockWithTimeMixing,
                ),
            ):
                x = layer(
                    x,
                    context,
                    time_context,
                    num_video_frames,
                    image_only_indicator,
                )
            elif isinstance(
                module,
                (
                    SpatialTemporalTransformerWith3DAttention,
                    SpatialTemporalTransformerWith3DAttentionAndPostHocViewAttentionMixing,
                ),
            ):
                x = layer(x, context, num_video_frames)
            elif isinstance(
                module,
                SpatialTransformer,
            ):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self,
        channels,
        use_conv,
        dims=2,
        out_channels=None,
        padding=1,
        third_up=False,
        kernel_size: int = 3,
        scale_factor: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.third_up = third_up
        self.scale_factor = scale_factor
        if use_conv:
            self.conv = conv_nd(
                dims, self.channels, self.out_channels, kernel_size, padding=padding
            )

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            t_factor = 1 if not self.third_up else self.scale_factor
            x = F.interpolate(
                x,
                (
                    t_factor * x.shape[2],
                    x.shape[3] * self.scale_factor,
                    x.shape[4] * self.scale_factor,
                ),
                mode="nearest",
            )
        else:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self, channels, use_conv, dims=2, out_channels=None, padding=1, third_down=False
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else ((1, 2, 2) if not third_down else (2, 2, 2))
        if use_conv:
            logpy.info(f"Building a Downsample layer with {dims} dims.")
            logpy.info(
                f"  --> settings are: \n in-chn: {self.channels}, out-chn: {self.out_channels}, "
                f"kernel-size: 3, stride: {stride}, padding: {padding}"
            )
            if dims == 3:
                logpy.info(f"  --> Downsampling third axis (time): {third_down}")
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 24):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
        use_pe: bool = False,
        pe_max_len: int = 24,
        pe_dropout: float = 0.0,
        num_video_frames: int = None,
        local_pos_emb_config: Optional[dict] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        if num_video_frames and use_pe:
            context_dim = query_dim  # always do self-attn for temporal attn
        else:
            context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.num_video_frames = num_video_frames

        self.pos_encoder = (
            PositionalEncoding(query_dim, dropout=pe_dropout, max_len=pe_max_len)
            if use_pe
            else None
        )

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.backend = backend

        self.local_pos_emb = None
        if local_pos_emb_config:
            if "params" in local_pos_emb_config:
                local_pos_emb_config["params"]["head_dim"] = dim_head
            else:
                local_pos_emb_config["params"] = {"head_dim": dim_head}
            self.local_pos_emb = instantiate_from_config(local_pos_emb_config)

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
        context_length: Optional[
            int
        ] = None,  # note: this is prepended context when using only sel-attention,
        # not context for cross attention
    ):
        h = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        if self.num_video_frames:
            _, d, _ = x.shape
            x = rearrange(x, "(b t) d c -> (b d) t c", t=self.num_video_frames)

        if self.pos_encoder is not None:
            x = self.pos_encoder(x)

        q = self.to_q(x)
        if self.num_video_frames and self.pos_encoder is not None:
            context = x  # always do self-attn for temporal attn
        else:
            context = default(context, x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if self.local_pos_emb is not None:
            q = self.local_pos_emb(q, context_length=context_length)
            k = self.local_pos_emb(k, context_length=context_length)
            v = self.local_pos_emb.potentially_reshape_v(v)

        ## old
        """
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        """
        ## new
        with sdp_kernel(**BACKEND_MAP[self.backend]):
            # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # scale is dim_head ** -0.5 per default

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]

        out = self.to_out(out)

        if self.num_video_frames:
            out = rearrange(out, "(b d) t c -> (b t) d c", d=d)

        return out


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_pe: bool = False,
        pe_max_len: int = 24,
        pe_dropout: float = 0.0,
        num_video_frames: int = None,
        local_pos_emb_config: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        logpy.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, "
            f"context_dim is {context_dim} and using {heads} heads with a "
            f"dimension of {dim_head}."
        )
        inner_dim = dim_head * heads
        if num_video_frames and use_pe:
            context_dim = query_dim  # always do self-attn for temporal attn
        else:
            context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head
        self.num_video_frames = num_video_frames

        self.pos_encoder = (
            PositionalEncoding(query_dim, dropout=pe_dropout, max_len=pe_max_len)
            if use_pe
            else None
        )

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

        self.local_pos_emb = None
        if local_pos_emb_config:
            if "params" in local_pos_emb_config:
                local_pos_emb_config["params"]["head_dim"] = dim_head
            else:
                local_pos_emb_config["params"] = {"head_dim": dim_head}
            local_pos_emb_config["params"]["out_reshape"] = "B H L D -> B L H D"
            self.local_pos_emb = instantiate_from_config(local_pos_emb_config)

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
        context_length: Optional[int] = None,
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        if self.num_video_frames:
            _, d, _ = x.shape
            x = rearrange(x, "(b t) d c -> (b d) t c", t=self.num_video_frames)

        if self.pos_encoder is not None:
            x = self.pos_encoder(x)

        q = self.to_q(x)
        context = default(context, x)
        if self.num_video_frames and self.pos_encoder is not None:
            context = x  # always do self-attn for temporal attn
        else:
            context = default(context, x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        if self.local_pos_emb is not None:
            # apply rotary embeddings, if specified, else no-op
            q = self.local_pos_emb(q, context_length=context_length)
            k = self.local_pos_emb(k, context_length=context_length)
            v = self.local_pos_emb.potentially_reshape_v(v)

        # actually compute the attention, what we cannot get enough of
        if version.parse(xformers.__version__) >= version.parse("0.0.21"):
            # NOTE: workaround for
            # https://github.com/facebookresearch/xformers/issues/845
            max_bs = 32768
            N = q.shape[0]
            n_batches = math.ceil(N / max_bs)
            out = list()
            for i_batch in range(n_batches):
                batch = slice(i_batch * max_bs, (i_batch + 1) * max_bs)
                out.append(
                    xformers.ops.memory_efficient_attention(
                        q[batch],
                        k[batch],
                        v[batch],
                        attn_bias=None,
                        op=self.attention_op,
                    )
                )
            out = torch.cat(out, 0)
        else:
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )

        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        out = self.to_out(out)

        if self.num_video_frames:
            out = rearrange(out, "(b d) t c -> (b t) d c", d=d)

        return out


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,  # ampere
    }

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        gated_ff: bool = True,
        checkpoint: bool = True,
        disable_self_attn: bool = False,
        attn_mode: str = "softmax",
        sdp_backend: Optional[SDPBackend] = None,
        use_pe: bool = False,
        pe_max_len: int = 24,
        pe_dropout: float = 0.0,
        num_video_frames: int = None,
        local_pos_emb_config: Optional[dict] = None,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            logpy.warn(
                f"Attention mode '{attn_mode}' is not available. Falling "
                f"back to native attention. This is not a problem in "
                f"Pytorch >= 2.0. FYI, you are running with PyTorch "
                f"version {torch.__version__}."
            )
            attn_mode = "softmax"
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            logpy.warn(
                "We do not support vanilla attention anymore, as it is too "
                "expensive. Sorry."
            )
            if not XFORMERS_IS_AVAILABLE:
                assert (
                    False
                ), "Please install xformers via e.g. 'pip install xformers==0.0.16'"
            else:
                logpy.info("Falling back to xformers efficient attention.")
                attn_mode = "softmax-xformers"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            backend=sdp_backend,
            use_pe=use_pe,
            pe_max_len=pe_max_len,
            pe_dropout=pe_dropout,
            num_video_frames=num_video_frames,
            local_pos_emb_config=local_pos_emb_config,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
            use_pe=use_pe,
            pe_max_len=pe_max_len,
            pe_dropout=pe_dropout,
            num_video_frames=num_video_frames,
            local_pos_emb_config=local_pos_emb_config,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        if self.checkpoint:
            logpy.debug(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.checkpoint:
            return checkpoint(self._forward, x, context)
        else:
            return self._forward(x, context)

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
            )
            + x
        )
        x = (
            self.attn2(
                self.norm2(x),
                context=context,
            )
            + x
        )
        x = self.ff(self.norm3(x)) + x
        return x


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        kernel_size=3,
        exchange_temb_dims=False,
        skip_t_emb=False,
        is_res=True,
        # dense embed
        use_dense_emb=False,
        dense_in_channels=None,
        use_dense_scale_shift_norm=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims
        # dense embed
        self.use_dense_emb = use_dense_emb
        self.use_dense_scale_shift_norm = use_dense_scale_shift_norm

        if isinstance(kernel_size, Iterable):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = (
            2 * self.out_channels if use_scale_shift_norm else self.out_channels
        )
        if self.skip_t_emb:
            logpy.info(f"Skipping timestep embedding in {self.__class__.__name__}")
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    self.emb_out_channels,
                ),
            )
        if self.use_dense_emb:
            dense_emb_channels = (
                2 * self.channels if use_dense_scale_shift_norm else self.channels
            )
            # https://github.com/lyndonzheng/Free3D/blob/04fdf2cec17d25e01d248bf8594bb12904578ca0/modules/diffusionmodules/openaimodel.py#L247
            self.dense_emb_layers = nn.Sequential(
                # nn.SiLU(),
                zero_module(
                    conv_nd(
                        dims,
                        dense_in_channels,
                        dense_emb_channels,
                        kernel_size=1,
                        padding=0,
                    )
                ),
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    dims,
                    self.out_channels,
                    self.out_channels,
                    kernel_size,
                    padding=padding,
                )
            ),
        )

        self.is_res = is_res
        if is_res:
            if self.out_channels == channels:
                self.skip_connection = nn.Identity()
            elif use_conv:
                self.skip_connection = conv_nd(
                    dims, channels, self.out_channels, kernel_size, padding=padding
                )
            else:
                self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor, dense_embed: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.use_checkpoint:
            return checkpoint(self._forward, x, emb, dense_embed)
        else:
            return self._forward(x, emb, dense_embed)

    def _forward(
        self, x: torch.Tensor, emb: torch.Tensor, dense_emb: torch.Tensor = None
    ) -> torch.Tensor:
        in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
        h = in_rest(x)

        # --- dense embed
        if self.use_dense_emb:
            assert dense_emb is not None
            new_size = h.shape[2:]
            mode = "bilinear" if len(new_size) == 2 else "trilinear"
            dense = self.dense_emb_layers(
                F.interpolate(dense_emb, size=new_size, mode=mode, align_corners=True)
            ).type(h.dtype)
            if self.use_dense_scale_shift_norm:
                dense_scale, dense_shift = torch.chunk(dense, 2, dim=1)
                h = h * (1 + dense_scale) + dense_shift
            else:
                h = h + dense
        # --- end of dense embed

        if self.updown:
            h = self.h_upd(h)
            x = self.x_upd(x)
        h = in_conv(h)

        if self.skip_t_emb:
            emb_out = torch.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
            h = h + emb_out
            h = self.out_layers(h)
        if self.is_res:
            h = self.skip_connection(x) + h
        return h


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        disable_self_attn: bool = False,
        use_linear: bool = False,
        attn_type: str = "softmax",
        use_checkpoint: bool = True,
        sdp_backend: Optional[SDPBackend] = None,
        register_length: int = 0,
        concat_context: bool = False,
        use_pe: bool = False,
        pe_max_len: int = 24,
        pe_dropout: float = 0.0,
        num_video_frames: int = None,
        local_pos_emb_config: Optional[dict] = None,
    ):
        super().__init__()
        logpy.debug(
            f"constructing {self.__class__.__name__} of depth {depth} w/ "
            f"{in_channels} channels and {n_heads} heads."
        )

        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                logpy.warn(
                    f"{self.__class__.__name__}: Found context dims "
                    f"{context_dim} of depth {len(context_dim)}, which does not "
                    f"match the specified 'depth' of {depth}. Setting context_dim "
                    f"to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                    use_pe=use_pe,
                    pe_max_len=pe_max_len,
                    pe_dropout=pe_dropout,
                    num_video_frames=num_video_frames,
                    local_pos_emb_config=local_pos_emb_config,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear
        self.concat_context = concat_context
        self.register_length = register_length
        if self.register_length > 0:
            self.register = nn.Parameter(torch.randn(1, register_length, inner_dim))
            # from https://github.com/facebookresearch/dinov2/blob/da4b3825f0ed64b7398ace00c5062503811d0cff/dinov2/models/vision_transformer.py#L176
            nn.init.normal_(self.register, std=1e-6)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()

        if self.use_linear:
            x = self.proj_in(x)

        context_length = None
        if self.concat_context:
            assert exists(context)
            context = torch.cat(context, 1)
            context_length = context.shape[1]
            x = torch.cat((context, x), 1)
            context = [None]

        if self.register_length > 0:
            x = torch.cat(
                (repeat(self.register, "1 ... -> b ...", b=x.shape[0]), x),
                1,
            )

        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, context=context[i])

        if self.register_length > 0:
            x = x[:, self.register_length :, :]

        if self.concat_context:
            assert context_length is not None
            x = x[:, context_length:, :]

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class CrossAttentionTimeMix(CrossAttention):
    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
        context_length: Optional[
            int
        ] = None,  # note: this is prepended context when using only sel-attention,
        # not context for cross attention
        image_only_indicator: Optional[torch.Tensor] = None,
    ):
        h = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        linear_kwargs = dict()
        if image_only_indicator is not None:
            # for potential LORA tuning, will be input kwargs for to_q/k/v/out
            linear_kwargs["image_only_indicator"] = image_only_indicator

        if self.num_video_frames:
            _, d, _ = x.shape
            x = rearrange(x, "(b t) d c -> (b d) t c", t=self.num_video_frames)

        if self.pos_encoder is not None:
            x = self.pos_encoder(x)

        q = self.to_q(x, **linear_kwargs)
        if self.num_video_frames and self.pos_encoder is not None:
            context = x  # always do self-attn for temporal attn
        else:
            context = default(context, x)
        context = default(context, x)
        k = self.to_k(context, **linear_kwargs)
        v = self.to_v(context, **linear_kwargs)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if self.local_pos_emb is not None:
            q = self.local_pos_emb(q, context_length=context_length)
            k = self.local_pos_emb(k, context_length=context_length)
            v = self.local_pos_emb.potentially_reshape_v(v)

        ## old
        """
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        """
        ## new
        with sdp_kernel(**BACKEND_MAP[self.backend]):
            # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # scale is dim_head ** -0.5 per default

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]

        # out = self.to_out(out)
        for layer in self.to_out:
            if hasattr(layer, "module") and isinstance(layer.module, nn.Linear):
                out = layer(out, **linear_kwargs)
            else:
                out = layer(out)

        if self.num_video_frames:
            out = rearrange(out, "(b d) t c -> (b t) d c", d=d)

        return out


class MemoryEfficientCrossAttentionTimeMix(MemoryEfficientCrossAttention):
    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
        context_length: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        linear_kwargs = dict()
        if image_only_indicator is not None:
            # for potential LORA tuning, will be input kwargs for to_q/k/v/out
            linear_kwargs["image_only_indicator"] = image_only_indicator

        if self.num_video_frames:
            _, d, _ = x.shape
            x = rearrange(x, "(b t) d c -> (b d) t c", t=self.num_video_frames)

        if self.pos_encoder is not None:
            x = self.pos_encoder(x)

        q = self.to_q(x, **linear_kwargs)
        context = default(context, x)
        if self.num_video_frames and self.pos_encoder is not None:
            context = x  # always do self-attn for temporal attn
        else:
            context = default(context, x)
        context = default(context, x)
        k = self.to_k(context, **linear_kwargs)
        v = self.to_v(context, **linear_kwargs)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        if self.local_pos_emb is not None:
            # apply rotary embeddings, if specified, else no-op
            q = self.local_pos_emb(q, context_length=context_length)
            k = self.local_pos_emb(k, context_length=context_length)
            v = self.local_pos_emb.potentially_reshape_v(v)

        # actually compute the attention, what we cannot get enough of
        if version.parse(xformers.__version__) >= version.parse("0.0.21"):
            # NOTE: workaround for
            # https://github.com/facebookresearch/xformers/issues/845
            max_bs = 32768
            N = q.shape[0]
            n_batches = math.ceil(N / max_bs)
            out = list()
            for i_batch in range(n_batches):
                batch = slice(i_batch * max_bs, (i_batch + 1) * max_bs)
                out.append(
                    xformers.ops.memory_efficient_attention(
                        q[batch],
                        k[batch],
                        v[batch],
                        attn_bias=None,
                        op=self.attention_op,
                    )
                )
            out = torch.cat(out, 0)
        else:
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )

        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]

        # out = self.to_out(out)
        for layer in self.to_out:
            if hasattr(layer, "module") and isinstance(layer.module, nn.Linear):
                out = layer(out, **linear_kwargs)
            else:
                out = layer(out)

        if self.num_video_frames:
            out = rearrange(out, "(b d) t c -> (b t) d c", d=d)

        return out


class BasicTransformerTimeMixBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttentionTimeMix,
        "softmax-xformers": MemoryEfficientCrossAttentionTimeMix,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        timesteps=None,
        ff_in=False,
        inner_dim=None,
        is_res=True,  # whether to add residual connection
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        switch_temporal_ca_to_sa=False,
    ):
        super().__init__()

        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        assert int(n_heads * d_head) == inner_dim

        self.is_res = (inner_dim == dim) and is_res

        if self.ff_in:
            self.norm_in = nn.LayerNorm(dim)
            self.ff_in = FeedForward(
                dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff
            )

        self.timesteps = timesteps
        self.disable_self_attn = disable_self_attn
        if self.disable_self_attn:
            self.attn1 = attn_cls(
                query_dim=inner_dim,
                heads=n_heads,
                dim_head=d_head,
                context_dim=context_dim,
                dropout=dropout,
            )  # is a cross-attention
        else:
            self.attn1 = attn_cls(
                query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
            )  # is a self-attention

        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff)
        self.ff.net[-1] = zero_module(self.ff.net[-1])  # zero-init the final layer

        if disable_temporal_crossattention:
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.attn2 = None
        else:
            self.norm2 = nn.LayerNorm(inner_dim)
            if switch_temporal_ca_to_sa:
                self.attn2 = attn_cls(
                    query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
                )  # is a self-attention
            else:
                self.attn2 = attn_cls(
                    query_dim=inner_dim,
                    context_dim=context_dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                )  # is self-attn if context is none

        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

        self.checkpoint = checkpoint
        if self.checkpoint:
            logpy.info(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None,
        timesteps: int = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.checkpoint:
            return checkpoint(self._forward, x, context, timesteps, **kwargs)
        else:
            return self._forward(x, context, timesteps=timesteps, **kwargs)

    def _forward(
        self,
        x,
        context=None,
        timesteps=None,
        **kwargs,
    ):
        assert self.timesteps or timesteps
        assert not (self.timesteps and timesteps) or self.timesteps == timesteps
        timesteps = self.timesteps or timesteps
        B, S, C = x.shape
        x = rearrange(x, "(b t) s c -> (b s) t c", t=timesteps)

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            x += x_skip

        if self.disable_self_attn:
            x = self.attn1(self.norm1(x), context=context, **kwargs) + x
        else:
            x = self.attn1(self.norm1(x), **kwargs) + x

        if self.attn2 is not None:
            if self.switch_temporal_ca_to_sa:
                x = self.attn2(self.norm2(x), **kwargs) + x
            else:
                x = self.attn2(self.norm2(x), context=context, **kwargs) + x
        x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        x = rearrange(
            x, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps
        )
        return x

    def get_last_layer(self):
        return self.ff.net[-1].weight


class SkipConnect(nn.Module):
    def __init__(
        self,
        rearrange_pattern: str = "b t -> (b t) 1 1",
    ):
        super().__init__()
        self.rearrange_pattern = rearrange_pattern

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if image_only_indicator is None:
            return x_spatial + x_temporal
        temporal_mask = rearrange(
            ~image_only_indicator.bool(), self.rearrange_pattern
        ).to(x_temporal.dtype)
        return x_spatial + temporal_mask * x_temporal


class AlphaBlender(nn.Module):
    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        rearrange_pattern: str = "b t -> (b t) 1 1",
    ):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.rearrange_pattern = rearrange_pattern

        assert (
            merge_strategy in self.strategies
        ), f"merge_strategy needs to be in {self.strategies}"

        def sigmoid_inverse(x, epsilon=1e-7):
            # Ensure values are in the range (0, 1) by clamping x
            x = torch.clamp(x, epsilon, 1 - epsilon)
            return torch.log(x / (1 - x))

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif (
            self.merge_strategy == "learned"
            or self.merge_strategy == "learned_with_images"
        ):
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(sigmoid_inverse(torch.Tensor([alpha])))
            )
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, image_only_indicator: torch.Tensor) -> torch.Tensor:
        # skip_time_mix = rearrange(repeat(skip_time_mix, 'b -> (b t) () () ()', t=t), '(b t) 1 ... -> b 1 t ...', t=t)
        if self.merge_strategy == "fixed":
            # make shape compatible
            # alpha = repeat(self.mix_factor, '1 -> b () t  () ()', t=t, b=bs)
            alpha = self.mix_factor
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)
            # make shape compatible
            # alpha = repeat(alpha, '1 -> s () ()', s = t * bs)
        elif self.merge_strategy == "learned_with_images":
            assert image_only_indicator is not None, "need image_only_indicator ..."
            alpha = torch.where(
                image_only_indicator.bool(),
                torch.ones(1, 1, device=image_only_indicator.device),
                rearrange(torch.sigmoid(self.mix_factor), "... -> ... 1"),
            )
            alpha = rearrange(alpha, self.rearrange_pattern)
            # make shape compatible
            # alpha = repeat(alpha, '1 -> s () ()', s = t * bs)
        else:
            raise NotImplementedError()
        return alpha

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        alpha = self.get_alpha(image_only_indicator)
        x = (
            alpha.to(x_spatial.dtype) * x_spatial
            + (1.0 - alpha).to(x_spatial.dtype) * x_temporal
        )
        return x


def get_alpha(
    merge_strategy: str,
    mix_factor: Optional[torch.Tensor],
    image_only_indicator: torch.Tensor,
    apply_sigmoid: bool = True,
    is_attn: bool = False,
) -> torch.Tensor:
    if merge_strategy == "fixed" or merge_strategy == "learned":
        alpha = mix_factor
    elif merge_strategy == "learned_with_images":
        alpha = torch.where(
            image_only_indicator.bool(),
            torch.ones(1, 1, device=image_only_indicator.device),
            rearrange(mix_factor, "... -> ... 1"),
        )
        if is_attn:
            alpha = rearrange(alpha, "b t -> (b t) 1 1")
        else:
            alpha = rearrange(alpha, "b t -> b 1 t 1 1")
    elif merge_strategy == "fixed_with_images":
        alpha = image_only_indicator
        if is_attn:
            alpha = rearrange(alpha, "b t -> (b t) 1 1")
        else:
            alpha = rearrange(alpha, "b t -> b 1 t 1 1")
    else:
        raise NotImplementedError
    return torch.sigmoid(alpha) if apply_sigmoid else alpha


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        )
        return a.reshape(bs, -1, length)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, **kwargs):
        # inputs = {"x": x}
        return checkpoint(self._forward, x)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class PostHocAttentionBlockWithTimeMixing(AttentionBlock):
    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        use_checkpoint: bool = False,
        use_new_attention_order: bool = False,
        dropout: float = 0.0,
        use_spatial_context: bool = False,
        merge_strategy: bool = "fixed",
        merge_factor: float = 0.5,
        apply_sigmoid_to_merge: bool = True,
        ff_in: bool = False,
        attn_mode: str = "softmax",
        disable_temporal_crossattention: bool = False,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            use_checkpoint=use_checkpoint,
            use_new_attention_order=use_new_attention_order,
        )
        inner_dim = n_heads * d_head

        self.time_mix_blocks = nn.ModuleList(
            [
                BasicTransformerTimeMixBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    checkpoint=use_checkpoint,
                    ff_in=ff_in,
                    attn_mode=attn_mode,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                )
            ]
        )
        self.in_channels = in_channels

        time_embed_dim = self.in_channels * 4
        self.time_mix_time_embed = nn.Sequential(
            linear(self.in_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, self.in_channels),
        )

        self.use_spatial_context = use_spatial_context

        if merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([merge_factor]))
        elif merge_strategy == "learned" or merge_strategy == "learned_with_images":
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(torch.Tensor([merge_factor]))
            )
        elif merge_strategy == "fixed_with_images":
            self.mix_factor = None
        else:
            raise ValueError(f"unknown merge strategy {merge_strategy}")

        self.get_alpha_fn = functools.partial(
            get_alpha,
            merge_strategy,
            self.mix_factor,
            apply_sigmoid=apply_sigmoid_to_merge,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ):
        if time_context is not None:
            raise NotImplementedError

        _, _, h, w = x.shape
        if exists(context):
            context = rearrange(context, "b t ... -> (b t) ...")
        if self.use_spatial_context:
            time_context = repeat(context[:, 0], "b ... -> (b n) ...", n=h * w)

        x = super().forward(
            x,
        )

        x = rearrange(x, "b c h w -> b (h w) c")
        x_mix = x

        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        num_frames = rearrange(num_frames, "b t -> (b t)")
        t_emb = timestep_embedding(num_frames, self.in_channels, repeat_only=False)
        emb = self.time_mix_time_embed(t_emb)
        emb = emb[:, None, :]
        x_mix = x_mix + emb

        x_mix = self.time_mix_blocks[0](
            x_mix, context=time_context, timesteps=timesteps
        )

        alpha = self.get_alpha_fn(image_only_indicator=image_only_indicator)
        x = alpha * x + (1.0 - alpha) * x_mix
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x


class PostHocSpatialTransformerWithTimeMixing(SpatialTransformer):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        use_spatial_context=False,
        timesteps=None,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        apply_sigmoid_to_merge: bool = True,
        time_context_dim=None,
        ff_in=False,
        checkpoint=False,
        time_depth=1,
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        time_mix_config: Optional[dict] = None,
        time_mix_legacy: bool = True,
        max_time_embed_period: int = 10000,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            attn_type=attn_mode,
            use_checkpoint=checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
        )
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period

        # ----------------- time_mixer ------------------
        self.time_mix_config = time_mix_config
        self.time_mix_legacy = time_mix_legacy
        if time_mix_config is not None:
            self.time_mixer = instantiate_from_config(time_mix_config)
        elif self.time_mix_legacy:
            if merge_strategy == "fixed":
                self.register_buffer("mix_factor", torch.Tensor([merge_factor]))
            elif merge_strategy == "learned" or merge_strategy == "learned_with_images":
                self.register_parameter(
                    "mix_factor", torch.nn.Parameter(torch.Tensor([merge_factor]))
                )
            elif merge_strategy == "fixed_with_images":
                self.mix_factor = None
            else:
                raise ValueError(f"unknown merge strategy {merge_strategy}")

            self.get_alpha_fn = partial(
                get_alpha,
                merge_strategy,
                self.mix_factor,
                apply_sigmoid=apply_sigmoid_to_merge,
                is_attn=True,
            )
        else:
            self.time_mixer = AlphaBlender(
                alpha=merge_factor, merge_strategy=merge_strategy
            )

        # ----------------- time_mix_blocks ------------------
        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        inner_dim = n_heads * d_head
        if use_spatial_context:
            time_context_dim = context_dim

        self.time_mix_blocks = nn.ModuleList(
            [
                BasicTransformerTimeMixBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=time_context_dim,
                    timesteps=timesteps,
                    checkpoint=checkpoint,
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    is_res=(
                        self.time_mix_legacy
                        or not isinstance(self.time_mixer, SkipConnect)
                    ),  # if it's a skip connect, we don't need to skip again in the block
                    attn_mode=attn_mode,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                )
                for _ in range(self.depth)
            ]
        )

        assert len(self.time_mix_blocks) == len(self.transformer_blocks)

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = self.in_channels * 4
        self.time_mix_time_embed = nn.Sequential(
            linear(self.in_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, self.in_channels),
        )
        nn.init.zeros_(self.time_mix_time_embed[-1].weight)
        nn.init.zeros_(self.time_mix_time_embed[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, _, h, w = x.shape
        x_in = x
        spatial_context = None
        if exists(context):
            spatial_context = context

        if self.use_spatial_context:
            assert (
                context.ndim == 3
            ), f"n dims of spatial context should be 3 but are {context.ndim}"

            time_context = context
            time_context_first_timestep = time_context[::timesteps]
            time_context = repeat(
                time_context_first_timestep, "b ... -> (b n) ...", n=h * w
            )
        elif time_context is not None and not self.use_spatial_context:
            time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            if time_context.ndim == 2:
                time_context = rearrange(time_context, "b c -> b 1 c")

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.use_linear:
            x = self.proj_in(x)

        if self.time_mix_legacy:
            alpha = self.get_alpha_fn(image_only_indicator=image_only_indicator)

        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        num_frames = rearrange(num_frames, "b t -> (b t)")
        t_emb = timestep_embedding(
            num_frames,
            self.in_channels,
            repeat_only=False,
            max_period=self.max_time_embed_period,
        )
        emb = self.time_mix_time_embed(t_emb)
        emb = emb[:, None, :]

        for it_, (block, mix_block) in enumerate(
            zip(self.transformer_blocks, self.time_mix_blocks)
        ):
            x = block(
                x,
                context=spatial_context,
            )

            x_mix = x
            x_mix = x_mix + emb

            x_mix = mix_block(x_mix, context=time_context, timesteps=timesteps)
            if self.time_mix_legacy:
                x = alpha.to(x.dtype) * x + (1.0 - alpha).to(x.dtype) * x_mix
            else:
                x = self.time_mixer(
                    x_spatial=x,
                    x_temporal=x_mix,
                    image_only_indicator=image_only_indicator,
                )
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out


class LegacyAlphaBlenderWithBug(nn.Module):
    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        rearrange_pattern: str = "b t -> (b t) 1 1",
    ):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.rearrange_pattern = rearrange_pattern

        assert (
            merge_strategy in self.strategies
        ), f"merge_strategy needs to be in {self.strategies}"

        def sigmoid_inverse(x, epsilon=1e-7):
            # Ensure values are in the range (0, 1) by clamping x
            x = torch.clamp(x, epsilon, 1 - epsilon)
            return torch.log(x / (1 - x))

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        elif (
            self.merge_strategy == "learned"
            or self.merge_strategy == "learned_with_images"
        ):
            self.register_parameter(
                "mix_factor", torch.nn.Parameter(sigmoid_inverse(torch.Tensor([alpha])))
            )
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, image_only_indicator: torch.Tensor) -> torch.Tensor:
        # skip_time_mix = rearrange(repeat(skip_time_mix, 'b -> (b t) () () ()', t=t), '(b t) 1 ... -> b 1 t ...', t=t)
        if self.merge_strategy == "fixed":
            # make shape compatible
            # alpha = repeat(self.mix_factor, '1 -> b () t  () ()', t=t, b=bs)
            alpha = self.mix_factor
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)
            # make shape compatible
            # alpha = repeat(alpha, '1 -> s () ()', s = t * bs)
        elif self.merge_strategy == "learned_with_images":
            assert image_only_indicator is not None, "need image_only_indicator ..."
            alpha = torch.where(
                image_only_indicator.bool(),
                torch.zeros(1, 1, device=image_only_indicator.device),
                rearrange(torch.sigmoid(self.mix_factor), "... -> ... 1"),
            )
            alpha = rearrange(alpha, self.rearrange_pattern)
            # make shape compatible
            # alpha = repeat(alpha, '1 -> s () ()', s = t * bs)
        else:
            raise NotImplementedError()
        return alpha

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_temporal: torch.Tensor,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        alpha = self.get_alpha(image_only_indicator)
        # NOTE: grave warning: this is a bug and only to replicate past behaviour
        x = (
            alpha.to(x_spatial.dtype) * x_temporal
            + (1.0 - alpha).to(x_spatial.dtype) * x_spatial
        )
        return x


class PostHocResBlockWithTime(ResBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        time_kernel_size: Union[int, List[int]] = 3,
        merge_strategy: bool = "fixed",
        merge_factor: float = 0.5,
        apply_sigmoid_to_merge: bool = True,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        time_mix_config: Optional[Dict] = None,
        time_mix_legacy: bool = True,
        replicate_bug: bool = False,
        # dense embed
        use_dense_emb=False,
        dense_in_channels=None,
        use_dense_scale_shift_norm=False,
    ):
        super().__init__(
            channels,
            emb_channels,
            dropout,
            out_channels=out_channels,
            use_conv=use_conv,
            use_scale_shift_norm=use_scale_shift_norm,
            dims=dims,
            use_checkpoint=use_checkpoint,
            up=up,
            down=down,
            use_dense_emb=use_dense_emb,
            dense_in_channels=dense_in_channels,
            use_dense_scale_shift_norm=use_dense_scale_shift_norm,
        )

        # ----------------- time_mixer ------------------
        self.time_mix_config = deepcopy(
            time_mix_config
        )  # cus we may update this config later
        self.time_mix_legacy = time_mix_legacy
        if self.time_mix_config is not None:
            self.time_mix_config.setdefault("params", {}).update(
                {"rearrange_pattern": "b t -> b 1 t 1 1"}
            )
            self.time_mixer = instantiate_from_config(self.time_mix_config)
        elif self.time_mix_legacy:
            if merge_strategy == "fixed":
                self.register_buffer("mix_factor", torch.Tensor([merge_factor]))
            elif merge_strategy == "learned" or merge_strategy == "learned_with_images":
                self.register_parameter(
                    "mix_factor", torch.nn.Parameter(torch.Tensor([merge_factor]))
                )
            elif merge_strategy == "fixed_with_images":
                self.mix_factor = None
            else:
                raise ValueError(f"unknown merge strategy {merge_strategy}")

            self.get_alpha_fn = functools.partial(
                get_alpha,
                merge_strategy,
                self.mix_factor,
                apply_sigmoid=apply_sigmoid_to_merge,
            )
        else:
            if replicate_bug:
                logpy.warning(
                    "*****************************************************************************************\n"
                    "GRAVE WARNING: YOU'RE USING THE BUGGY LEGACY ALPHABLENDER!!! ARE YOU SURE YOU WANT THIS?!\n"
                    "*****************************************************************************************"
                )
                self.time_mixer = LegacyAlphaBlenderWithBug(
                    alpha=merge_factor,
                    merge_strategy=merge_strategy,
                    rearrange_pattern="b t -> b 1 t 1 1",
                )
            else:
                self.time_mixer = AlphaBlender(
                    alpha=merge_factor,
                    merge_strategy=merge_strategy,
                    rearrange_pattern="b t -> b 1 t 1 1",
                )

        # ----------------- time_mix_blocks ------------------
        self.time_mix_blocks = ResBlock(
            default(out_channels, channels),
            emb_channels,
            dropout=dropout,
            dims=3,
            out_channels=default(out_channels, channels),
            use_scale_shift_norm=False,
            use_conv=False,
            up=False,
            down=False,
            kernel_size=time_kernel_size,
            use_checkpoint=use_checkpoint,
            exchange_temb_dims=True,
            is_res=(
                time_mix_legacy or not isinstance(self.time_mixer, SkipConnect)
            ),  # if it's a skip connect, we don't need to skip again in the block
            use_dense_emb=use_dense_emb,
            dense_in_channels=dense_in_channels,
            use_dense_scale_shift_norm=use_dense_scale_shift_norm,
        )

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        dense_emb: torch.Tensor,
        num_video_frames: int,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = super().forward(x, emb, dense_emb)

        x_mix = rearrange(x, "(b t) c h w -> b c t h w", t=num_video_frames)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=num_video_frames)

        x = self.time_mix_blocks(
            x,
            rearrange(emb, "(b t) ... -> b t ...", t=num_video_frames),
            rearrange(dense_emb, "(b t) c ... -> b c t ...", t=num_video_frames)
            if dense_emb is not None
            else None,
        )

        if self.time_mix_legacy:
            alpha = self.get_alpha_fn(image_only_indicator=image_only_indicator)
            x = alpha.to(x.dtype) * x + (1.0 - alpha).to(x.dtype) * x_mix
        else:
            x = self.time_mixer(
                x_spatial=x_mix, x_temporal=x, image_only_indicator=image_only_indicator
            )
        x = rearrange(x, "b c t h w -> (b t) c h w")
        return x


class PatchedOutLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        dims: int = 2,
        num_layers: int = 1,
        original_channels: int = 4,
    ):
        super().__init__()
        self.och = original_channels
        ch = channels - original_channels
        self.ch = ch
        out_ch = out_channels - original_channels

        layers = []
        for _ in range(num_layers - 1):
            layers.append(normalization(ch))
            layers.append(nn.SiLU())
            layers.append(zero_module(conv_nd(dims, ch, ch, 3, padding=1)))
        layers.append(normalization(ch))
        layers.append(nn.SiLU())
        layers.append(zero_module(conv_nd(dims, ch, out_ch, 3, padding=1)))
        self.patched = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        ox, x = torch.split(x, [self.och, self.ch], dim=1)
        return torch.cat([ox, self.patched(x)], dim=1)


class SpatialUNetModelWithTime(nn.Module):
    """
    in_channels: 11
    model_channels: 320
    out_channels: 4
    num_res_blocks: 2
    attention_resolutions: [4, 2, 1]
    dropout: 0.0
    channel_mult: [1, 2, 4, 4]
    conv_resample: True
    dims: 2
    num_classes: None
    use_checkpoint: False
    num_heads: -1
    num_head_channels: 64
    num_heads_upsample: -1
    use_scale_shift_norm: False
    resblock_updown: False
    use_new_attention_order: False
    use_spatial_transformer: True
    transformer_depth: 1
    transformer_depth_middle: None
    context_dim: 1024
    time_downup: False
    time_context_dim: None
    extra_ff_mix_layer: True
    use_spatial_context: True
    time_block_merge_strategy: fixed
    time_block_merge_factor: 0.5
    spatial_transformer_attn_type: softmax-xformers
    time_kernel_size: 3
    use_linear_in_transformer: True
    legacy: False
    adm_in_channels: None
    use_temporal_resblock: True
    disable_temporal_crossattention: False
    time_mix_config: {'target': 'sgm.modules.diffusionmodules.util.SkipConnect'}
    time_mix_legacy: True
    max_ddpm_temb_period: 10000
    replicate_time_mix_bug: False
    use_dense_embed: True
    dense_in_channels: 6
    use_dense_scale_shift_norm: True
    extra_out_layers: 0
    original_out_channels: 4
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: int,
        dropout: float = 0.0,
        channel_mult: List[int] = [1, 2, 4, 8],
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        use_spatial_transformer: bool = False,
        transformer_depth: Union[List[int], int] = 1,
        transformer_depth_middle: Optional[int] = None,
        context_dim: Optional[int] = None,
        time_downup: bool = False,
        time_context_dim: Optional[int] = None,
        extra_ff_mix_layer: bool = False,
        use_spatial_context: bool = False,
        time_block_merge_strategy: str = "fixed",
        time_block_merge_factor: float = 0.5,
        spatial_transformer_attn_type: str = "softmax",
        time_kernel_size: Union[int, List[int]] = 3,
        use_linear_in_transformer: bool = False,
        legacy: bool = True,
        adm_in_channels: Optional[int] = None,
        use_temporal_resblock: bool = True,
        disable_temporal_crossattention: bool = False,
        time_mix_config: Optional[Dict] = None,
        time_mix_legacy: bool = True,
        max_ddpm_temb_period: int = 10000,
        replicate_time_mix_bug: bool = False,
        # dense embed
        use_dense_embed: bool = False,
        dense_in_channels: Optional[int] = None,
        use_dense_scale_shift_norm: bool = False,
        # extra out layer for new modalities
        extra_out_layers: int = 0,
        original_out_channels: int = 4,  # original 4 channels for rgb modality
    ):
        super().__init__()

        if use_spatial_transformer:
            assert context_dim is not None

        if context_dim is not None:
            assert use_spatial_transformer

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1

        if num_head_channels == -1:
            assert num_heads != -1

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = default(
            transformer_depth_middle, transformer_depth[-1]
        )

        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.use_spatial_transformer = use_spatial_transformer
        self.spatial_transformer_attn_type = spatial_transformer_attn_type
        self.use_spatial_context = use_spatial_context

        self.time_block_merge_strategy = time_block_merge_strategy
        self.time_block_merge_factor = time_block_merge_factor
        self.time_context_dim = time_context_dim
        self.extra_ff_mix_layer = extra_ff_mix_layer
        self.use_linear_in_transformer = use_linear_in_transformer
        self.disable_temporal_crossattention = disable_temporal_crossattention
        self.time_mix_config = time_mix_config
        self.time_mix_legacy = time_mix_legacy
        self.max_ddpm_temb_period = max_ddpm_temb_period
        self.use_temporal_resblocks = use_temporal_resblock
        self.time_kernel_size = time_kernel_size
        self.use_new_attention_order = use_new_attention_order
        self.replicate_time_mix_bug = replicate_time_mix_bug

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )

            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    self.get_resblock(
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_dense_emb=use_dense_embed,
                        dense_in_channels=dense_in_channels,
                        use_dense_scale_shift_norm=use_dense_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )

                    layers.append(
                        self.get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"input_ds{ds}",
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        self.get_resblock(
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            use_dense_emb=use_dense_embed,
                            dense_in_channels=dense_in_channels,
                            use_dense_scale_shift_norm=use_dense_scale_shift_norm,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_down=time_downup,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)

                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        self.middle_block = TimestepEmbedSequential(
            self.get_resblock(
                ch=ch,
                time_embed_dim=time_embed_dim,
                out_ch=None,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_dense_emb=use_dense_embed,
                dense_in_channels=dense_in_channels,
                use_dense_scale_shift_norm=use_dense_scale_shift_norm,
            ),
            self.get_attention_layer(
                ch,
                num_heads,
                dim_head,
                name=f"middle_ds{ds}",
                depth=transformer_depth_middle,
                context_dim=context_dim,
                use_checkpoint=use_checkpoint,
            ),
            self.get_resblock(
                ch=ch,
                out_ch=None,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_dense_emb=use_dense_embed,
                dense_in_channels=dense_in_channels,
                use_dense_scale_shift_norm=use_dense_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    self.get_resblock(
                        ch=ch + ich,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_dense_emb=use_dense_embed,
                        dense_in_channels=dense_in_channels,
                        use_dense_scale_shift_norm=use_dense_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )

                    layers.append(
                        self.get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"output_ds{ds}",
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    layers.append(
                        self.get_resblock(
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            use_dense_emb=use_dense_embed,
                            dense_in_channels=dense_in_channels,
                            use_dense_scale_shift_norm=use_dense_scale_shift_norm,
                        )
                        if resblock_updown
                        else Upsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_up=time_downup,
                        )
                    )

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        out = self.get_out_layer(
            ch,
            out_channels,
            dims,
            extra_out_ly=extra_out_layers,
            original_out_ch=original_out_channels,
        )
        self.out = nn.Sequential(*out)

    def get_attention_layer(
        self,
        ch,
        num_heads,
        dim_head,
        name,
        depth=1,
        context_dim=None,
        use_checkpoint=False,
        disabled_sa=False,
    ):
        if self.use_spatial_transformer:
            return PostHocSpatialTransformerWithTimeMixing(
                ch,
                num_heads,
                dim_head,
                depth=depth,
                context_dim=context_dim,
                time_context_dim=self.time_context_dim,
                dropout=self.dropout,
                ff_in=self.extra_ff_mix_layer,
                use_spatial_context=self.use_spatial_context,
                merge_strategy=self.time_block_merge_strategy,
                merge_factor=self.time_block_merge_factor,
                checkpoint=use_checkpoint,
                use_linear=self.use_linear_in_transformer,
                attn_mode=self.spatial_transformer_attn_type,
                disable_self_attn=disabled_sa,
                disable_temporal_crossattention=self.disable_temporal_crossattention,
                time_mix_config=self.time_mix_config,
                time_mix_legacy=self.time_mix_legacy,
                max_time_embed_period=self.max_ddpm_temb_period,
            )
        else:
            assert False, "`PostHocAttentionBlockWithTimeMixing` not supported anymore"

    def get_resblock(
        self,
        ch,
        time_embed_dim,
        dropout,
        out_ch,
        dims,
        use_checkpoint,
        use_scale_shift_norm,
        down=False,
        up=False,
        use_dense_emb=False,
        dense_in_channels=None,
        use_dense_scale_shift_norm=False,
    ):
        if self.use_temporal_resblocks:
            return PostHocResBlockWithTime(
                merge_factor=self.time_block_merge_factor,
                merge_strategy=self.time_block_merge_strategy,
                time_kernel_size=self.time_kernel_size,
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=out_ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                down=down,
                up=up,
                time_mix_config=self.time_mix_config,
                time_mix_legacy=self.time_mix_legacy,
                replicate_bug=self.replicate_time_mix_bug,
                use_dense_emb=use_dense_emb,
                dense_in_channels=dense_in_channels,
                use_dense_scale_shift_norm=use_dense_scale_shift_norm,
            )
        else:
            return ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=out_ch,
                use_checkpoint=use_checkpoint,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                down=down,
                up=up,
                use_dense_emb=use_dense_emb,
                dense_in_channels=dense_in_channels,
                use_dense_scale_shift_norm=use_dense_scale_shift_norm,
            )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        dense_y: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        num_video_frames: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ):
        assert (
            (y is not None) == (self.num_classes is not None)
        ), "must specify y if and only if the model is class-conditional -> no, relax this TODO"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(
                h,
                emb,
                context=context,
                dense_emb=dense_y,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames,
            )
            hs.append(h)
        h = self.middle_block(
            h,
            emb,
            context=context,
            dense_emb=dense_y,
            image_only_indicator=image_only_indicator,
            time_context=time_context,
            num_video_frames=num_video_frames,
        )
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(
                h,
                emb,
                context=context,
                dense_emb=dense_y,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames,
            )
        h = h.type(x.dtype)
        return self.out(h)

    def get_out_layer(
        self,
        ch,
        out_ch,
        dims,
        extra_out_ly,
        original_out_ch,
    ):
        if extra_out_ly > 0 and out_ch > original_out_ch:
            inter_ch = original_out_ch + self.model_channels
            out = (
                normalization(ch),
                nn.SiLU(),
                zero_module(conv_nd(dims, self.model_channels, inter_ch, 3, padding=1)),
                PatchedOutLayer(
                    inter_ch,
                    out_ch,
                    dims=dims,
                    num_layers=extra_out_ly,
                    original_channels=original_out_ch,
                ),
            )
        else:
            out = (
                normalization(ch),
                nn.SiLU(),
                zero_module(conv_nd(dims, self.model_channels, out_ch, 3, padding=1)),
            )
        return out


class SpatialTemporalTransformerWith3DAttention(SpatialTransformer):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        name,
        depth=1,
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        use_spatial_context=False,
        checkpoint=False,
        attn_mode="softmax",
        disable_self_attn=False,
        unflatten_names=[],
        **kwargs,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            attn_type=attn_mode,
            use_checkpoint=checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
            **kwargs,
        )
        self.name = name
        self.unflatten_names = unflatten_names

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
    ) -> torch.Tensor:
        _, _, h, w = x.shape
        x_in = x

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)

        if self.name in self.unflatten_names:
            x = rearrange(x, "(b t) c h w -> b (t h w) c", t=timesteps)
            # we assume here context is copied over temporal dimension which usually holds true
            context = context[::timesteps]
        else:
            x = rearrange(x, "b c h w -> b (h w) c")

        if self.use_linear:
            x = self.proj_in(x)

        for it_, block in enumerate(self.transformer_blocks):
            x = block(x, context=context)

        if self.use_linear:
            x = self.proj_out(x)

        if self.name in self.unflatten_names:
            x = rearrange(x, "b (t h w) c -> (b t) c h w", t=timesteps, h=h, w=w)
        else:
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out


class SpatialTemporalTransformerWith3DAttentionAndPostHocViewAttentionMixing(
    SpatialTransformer
):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        name,
        depth=1,
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        use_spatial_context=False,
        checkpoint=False,
        attn_mode="softmax",
        disable_self_attn=False,
        unflatten_names=[],
        #
        timesteps=None,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        time_context_dim=None,
        time_depth=1,
        ff_in=False,
        disable_temporal_crossattention=False,
        time_mix_config: Optional[dict] = None,
        #
        **kwargs,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            attn_type=attn_mode,
            use_checkpoint=checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
            **kwargs,
        )
        self.name = name
        self.unflatten_names = unflatten_names

        self.time_depth = time_depth
        self.depth = depth
        time_mix_d_head = d_head
        n_time_mix_heads = n_heads
        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)
        inner_dim = n_heads * d_head
        self.use_spatial_context = use_spatial_context
        if use_spatial_context:
            time_context_dim = context_dim

        # ----------------- time_mixer ------------------
        if time_mix_config is not None:
            self.time_mixer = instantiate_from_config(time_mix_config)
        else:
            self.time_mixer = AlphaBlender(
                alpha=merge_factor, merge_strategy=merge_strategy
            )

        # ----------------- time_mix_blocks ------------------
        self.time_mix_blocks = nn.ModuleList(
            [
                BasicTransformerTimeMixBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=time_context_dim,
                    timesteps=timesteps,
                    checkpoint=checkpoint,
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    is_res=(
                        not isinstance(self.time_mixer, SkipConnect)
                    ),  # if it's a skip connect, we don't need to skip again in the block
                    attn_mode=attn_mode,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                )
                for _ in range(self.depth)
            ]
        )
        assert len(self.time_mix_blocks) == len(self.transformer_blocks)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
    ) -> torch.Tensor:
        B, c, h, w = x.shape
        x_in = x

        if self.use_spatial_context:
            assert (
                context.ndim == 3
            ), f"n dims of spatial context should be 3 but are {context.ndim}"

            time_context = context
            time_context_first_timestep = time_context[::timesteps]
            time_context = repeat(
                time_context_first_timestep, "b ... -> (b n) ...", n=h * w
            )
        else:
            time_context = None

        if self.name in self.unflatten_names:
            # we assume here context is copied over temporal dimension which usually holds true
            context = context[::timesteps]

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        if self.use_linear:
            x = self.proj_in(x)

        for it_, (block, mix_block) in enumerate(
            zip(self.transformer_blocks, self.time_mix_blocks)
        ):
            if self.name in self.unflatten_names:
                x = rearrange(x, "(b t) (h w) c -> b (t h w) c", t=timesteps, h=h, w=w)

            x = block(x, context=context)

            if self.name in self.unflatten_names:
                x = rearrange(x, "b (t h w) c -> (b t) (h w) c", t=timesteps, h=h, w=w)

            x_mix = mix_block(x, context=time_context, timesteps=timesteps)
            x = self.time_mixer(
                x_spatial=x,
                x_temporal=x_mix,
            )

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out


class SpatialTemporalTransformerWith3DAttentionAndPostHocTimeMixing(
    PostHocSpatialTransformerWithTimeMixing
):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        name,
        unflatten_names=[],
        time_mix_attn_for_image_only=False,
        time_mix_attn_for_image_only_without_pos_emb=True,
        time_mix_attn_lora=False,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            **kwargs,
        )
        self.name = name
        self.unflatten_names = unflatten_names
        self.time_mix_attn_for_image_only = time_mix_attn_for_image_only
        self.time_mix_attn_for_image_only_without_pos_emb = (
            time_mix_attn_for_image_only_without_pos_emb
        )
        if (
            self.time_mix_attn_for_image_only
            and self.time_mix_attn_for_image_only_without_pos_emb
        ):
            del self.time_mix_time_embed
        self.time_mix_attn_lora = time_mix_attn_lora

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, c, h, w = x.shape
        x_in = x
        spatial_context = None
        if exists(context):
            spatial_context = context

        if self.use_spatial_context:
            assert (
                context.ndim == 3
            ), f"n dims of spatial context should be 3 but are {context.ndim}"

            time_context = context
            time_context_first_timestep = time_context[::timesteps]
            time_context = repeat(
                time_context_first_timestep, "b ... -> (b n) ...", n=h * w
            )
        elif time_context is not None and not self.use_spatial_context:
            time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            if time_context.ndim == 2:
                time_context = rearrange(time_context, "b c -> b 1 c")

        if self.name in self.unflatten_names:
            # we assume here context is copied over temporal dimension which usually holds true
            spatial_context = spatial_context[::timesteps]

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.use_linear:
            x = self.proj_in(x)

        # -----------------------------------------------------------------
        # 1) if time_mix_attn_for_image_only == False (default)
        #    image_only_indicator_ = image_only_indicator => only using full model with ordered input
        # 2) if time_mix_attn_for_image_only == True
        #    image_only_indicator_ = False => always using full model
        # -----------------------------------------------------------------
        image_only_indicator_ = torch.logical_and(
            image_only_indicator.bool(),
            torch.full_like(
                image_only_indicator, not self.time_mix_attn_for_image_only
            ).bool(),
        ).to(image_only_indicator.dtype)

        if self.time_mix_legacy:
            alpha = self.get_alpha_fn(image_only_indicator=image_only_indicator_)

        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=B // timesteps)
        num_frames = rearrange(num_frames, "b t -> (b t)")

        if (
            not self.time_mix_attn_for_image_only
            or not self.time_mix_attn_for_image_only_without_pos_emb
        ):
            t_emb = timestep_embedding(
                num_frames,
                self.in_channels,
                repeat_only=False,
                max_period=self.max_time_embed_period,
            )
            emb = self.time_mix_time_embed(t_emb)
            emb = emb[:, None, :]

        for it_, (block, mix_block) in enumerate(
            zip(self.transformer_blocks, self.time_mix_blocks)
        ):
            if self.name in self.unflatten_names:
                x = rearrange(x, "(b t) (h w) c -> b (t h w) c", t=timesteps, h=h, w=w)

            x = block(x, context=spatial_context)

            if self.name in self.unflatten_names:
                x = rearrange(x, "b (t h w) c -> (b t) (h w) c", t=timesteps, h=h, w=w)

            # -----------------------------------------------------------------
            # 1) if time_mix_attn_for_image_only == False (default)
            #    emb_mask = True => always using pos emb
            # 2) if time_mix_attn_for_image_only == True
            #    2.1) if time_mix_attn_for_image_only_without_pos_emb == True (default)
            #         emb_mask = False => never using pos emb
            #    2.2) if time_mix_attn_for_image_only_without_pos_emb == False
            #         emb_mask = ~image_only_indicator => only using pos emb with ordered input
            # -----------------------------------------------------------------
            x_mix = x
            if not self.time_mix_attn_for_image_only:
                x_mix = x_mix + emb
            elif not self.time_mix_attn_for_image_only_without_pos_emb:
                x_mix = (
                    x_mix
                    + (~image_only_indicator.bool())
                    .to(image_only_indicator.dtype)
                    .flatten()[:, None, None]
                    * emb
                )

            x_mix = mix_block(
                x_mix,
                context=time_context,
                timesteps=timesteps,
                image_only_indicator=image_only_indicator
                if self.time_mix_attn_lora
                else None,
            )

            if self.time_mix_legacy:
                x = alpha.to(x.dtype) * x + (1.0 - alpha).to(x.dtype) * x_mix
            else:
                x = self.time_mixer(
                    x_spatial=x,
                    x_temporal=x_mix,
                    image_only_indicator=image_only_indicator_,
                )

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out


class _3DUNetModelWithViewAttn(SpatialUNetModelWithTime):
    """
    unflatten_names: ["middle_ds8", "output_ds4", "output_ds2"]
    in_channels: 11
    model_channels: 320
    out_channels: 4
    num_res_blocks: 2
    attention_resolutions: [4, 2, 1]
    dropout: 0.0
    channel_mult: [1, 2, 4, 4]
    conv_resample: True
    dims: 2
    num_classes: None
    use_checkpoint: False
    num_heads: -1
    num_head_channels: 64
    num_heads_upsample: -1
    use_scale_shift_norm: False
    resblock_updown: False
    use_new_attention_order: False
    use_spatial_transformer: True
    transformer_depth: 1
    transformer_depth_middle: None
    context_dim: 1024
    time_downup: False
    time_context_dim: None
    extra_ff_mix_layer: True
    use_spatial_context: True
    time_block_merge_strategy: fixed
    time_block_merge_factor: 0.5
    spatial_transformer_attn_type: softmax-xformers
    time_kernel_size: 3
    use_linear_in_transformer: True
    legacy: False
    adm_in_channels: None
    use_temporal_resblock: True
    disable_temporal_crossattention: False
    time_mix_config: {'target': 'sgm.modules.diffusionmodules.util.SkipConnect'}
    time_mix_legacy: True
    max_ddpm_temb_period: 10000
    replicate_time_mix_bug: False
    use_dense_embed: True
    dense_in_channels: 6
    use_dense_scale_shift_norm: True
    extra_out_layers: 0
    original_out_channels: 4
    """

    def __init__(self, *args, unflatten_names: List[str] = [], **kwargs):
        self.unflatten_names = unflatten_names
        super().__init__(*args, **kwargs)
        assert self.use_spatial_transformer, "_3DUNetModel requires spatial transformer"

    def get_attention_layer(
        self,
        ch,
        num_heads,
        dim_head,
        name,
        depth=1,
        context_dim=None,
        use_checkpoint=False,
        disabled_sa=False,
    ):
        return SpatialTemporalTransformerWith3DAttentionAndPostHocViewAttentionMixing(
            ch,
            num_heads,
            dim_head,
            name=name,
            depth=depth,
            context_dim=context_dim,
            time_context_dim=self.time_context_dim,
            dropout=self.dropout,
            ff_in=self.extra_ff_mix_layer,
            use_spatial_context=self.use_spatial_context,
            merge_strategy=self.time_block_merge_strategy,
            merge_factor=self.time_block_merge_factor,
            checkpoint=use_checkpoint,
            use_linear=self.use_linear_in_transformer,
            attn_mode=self.spatial_transformer_attn_type,
            disable_self_attn=disabled_sa,
            disable_temporal_crossattention=self.disable_temporal_crossattention,
            time_mix_config=self.time_mix_config,
            unflatten_names=self.unflatten_names,
        )

    def get_resblock(
        self,
        ch,
        time_embed_dim,
        dropout,
        out_ch,
        dims,
        use_checkpoint,
        use_scale_shift_norm,
        down=False,
        up=False,
        use_dense_emb=False,
        dense_in_channels=None,
        use_dense_scale_shift_norm=False,
    ):
        return ResBlock(
            channels=ch,
            emb_channels=time_embed_dim,
            dropout=dropout,
            out_channels=out_ch,
            use_checkpoint=use_checkpoint,
            dims=dims,
            use_scale_shift_norm=use_scale_shift_norm,
            down=down,
            up=up,
            use_dense_emb=use_dense_emb,
            dense_in_channels=dense_in_channels,
            use_dense_scale_shift_norm=use_dense_scale_shift_norm,
        )
