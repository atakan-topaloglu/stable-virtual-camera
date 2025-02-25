import copy
import os
import os.path as osp
import secrets
import time
from datetime import datetime
from pathlib import Path
from glob import glob

import gradio as gr
import httpx
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
import tyro
import viser
import viser.transforms as vt
from einops import rearrange, repeat
from gradio import networking
from gradio.tunneling import CERTIFICATE_PATH, Tunnel
from dataclasses import dataclass
from typing import Literal
from pytorch_lightning import seed_everything
from tqdm import tqdm
import imageio.v3 as iio
from PIL import Image
import tempfile

from stableviews.eval import (
    IS_TORCH_NIGHTLY,
    chunk_input_and_test,
    infer_prior_stats,
    run_one_scene,
    transform_img_and_K,
    update_kv_for_dict,
    get_unique_embedder_keys_from_conditioner,
    get_plucker_coordinates,
    to_hom_pose,
    get_k_from_dict,
    extend_dict,
    decode_output,
    do_sample,
    create_samplers,
)
from stableviews.geometry import (
    normalize_scene,
    get_default_intrinsics,
    get_panning_w2cs,
    get_roll_w2cs,
    get_lemniscate_w2cs,
    get_arc_horizontal_w2cs,
    generate_spiral_path,
    DEFAULT_FOV_RAD
)
from stableviews.gui import define_gui
from stableviews.model import SGMWrapper
from stableviews.modules.autoencoder import AutoEncoder
from stableviews.modules.conditioner import CLIPConditioner
from stableviews.modules.preprocessor import Dust3rPipeline
from stableviews.sampling import DDPMDiscretization, DiscreteDenoiser
from stableviews.utils import load_model

device = "cuda:0"

os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.environ.get("TMPDIR", "/tmp"), "gradio")

# Constants.
WORK_DIR = "work_dirs/demo_gr"
MAX_SESSIONS = 1
EXAMPLE_DIR = "assets/demo-assets/"
EXAMPLE_MAP = [
    (
        "assets/demo-assets/nonsquare_1.png",
        ["assets/demo-assets/nonsquare_1.png"],
    ),
    (
        "assets/demo-assets/scene_1.png",
        ["assets/demo-assets/scene_1.png"],
    ),
    (
        "assets/demo-assets/scene_2_1.png",
        [
            "assets/demo-assets/scene_2_1.png",
            "assets/demo-assets/scene_2_2.png",
            "assets/demo-assets/scene_2_3.png",
            "assets/demo-assets/scene_2_4.png",
        ],
    ),
]

# Delete previous gradio temp dir folder
if os.path.exists(os.environ["GRADIO_TEMP_DIR"]):
    print(f"Deleting {os.environ['GRADIO_TEMP_DIR']}")
    import shutil

    shutil.rmtree(os.environ["GRADIO_TEMP_DIR"])

# Precompute hash values for all example images.
EXAMPLE_HASHES = {}
for img_path in sorted(glob(f"{EXAMPLE_DIR}*png")):
    with Image.open(img_path) as img:
        np_img = np.array(img)
    img_hash = hashlib.sha256(np_img.tobytes()).hexdigest()[:16]
    EXAMPLE_HASHES[img_hash] = img_path

if IS_TORCH_NIGHTLY:
    COMPILE = True
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
else:
    COMPILE = False

# Shared global variables across sessions.
DUST3R = Dust3rPipeline(device=device)  # type: ignore
MODEL = SGMWrapper(load_model("stabilityai/stableviews", "model.safetensors", device="cpu", verbose=True).eval()).to(device)

AE = AutoEncoder(chunk_size=1).to(device)
CONDITIONER = CLIPConditioner().to(device)
DISCRETIZATION = DDPMDiscretization()
DENOISER = DiscreteDenoiser(discretization=DISCRETIZATION, num_idx=1000, device=device)
VERSION_DICT = {
    "H": 576,
    "W": 576,
    "T": 21,
    "C": 4,
    "f": 8,
    "options": {},
}
SERVERS = {}

if COMPILE:
    MODEL = torch.compile(MODEL, dynamic=False)
    CONDITIONER = torch.compile(CONDITIONER, dynamic=False)
    AE = torch.compile(AE, dynamic=False)


@dataclass
class StableViewsSingleImageConfig:
    context_window: int = 21
    target_wh: tuple[int, int] = (576, 576)
    output_root = "logs/_gradio/single_img/"
    chunk_strategy: Literal["nearest", "interp", "interp-gt"] = "interp-gt"
    chunk_strategy_first_pass: str = "gt_nearest"
    seed: int = 23

    def __post_init__(self):
        self.options = {
            "as_shuffled": True,
            "discretization": 0,
            "beta_linear_start": 5e-6,
            "log_snr_shift": 2.4,
            "guider_types": [1, 2],
            "cfg": (3.0, 2.0),
            "num_steps": 50,
            "num_frames": self.context_window,
            "encoding_t": 1,
            "decoding_t": 1,
        }


@dataclass
class StableViewsSingleImageData(object):
    input_imgs: torch.Tensor
    input_c2ws: torch.Tensor
    input_Ks: torch.Tensor
    target_c2ws: torch.Tensor
    target_Ks: torch.Tensor
    anchor_c2ws: torch.Tensor
    anchor_Ks: torch.Tensor
    num_inputs: int
    num_targets: int
    num_anchors: int
    input_indices: list[int]
    anchor_indices: list[int]
    target_indices: list[int]
    camera_scale: float = 2.0


class StableViewsSingleImageRenderer(object):
    def __init__(self, cfg: StableViewsSingleImageConfig):
        self.cfg = cfg
        self.engine = MODEL

    def get_value_dict(
        self,
        curr_imgs,
        curr_imgs_clip,
        curr_input_frame_indices,
        curr_c2ws,
        curr_Ks,
        curr_input_camera_indices,
        curr_depths,
        all_c2ws,
        camera_scale=2.0,
        as_shuffled=True,
    ):
        assert sorted(curr_input_camera_indices) == sorted(
            range(len(curr_input_camera_indices))
        )
        H, W, T, F = curr_imgs.shape[-2], curr_imgs.shape[-1], len(curr_imgs), 8

        value_dict = {}
        value_dict["image_only_indicator"] = int(as_shuffled)

        value_dict["cond_frames_without_noise"] = curr_imgs_clip[
            curr_input_frame_indices
        ]
        T = len(curr_imgs)
        value_dict["cond_frames_mask"] = torch.zeros(T, dtype=torch.bool)
        value_dict["cond_frames_mask"][curr_input_frame_indices] = True
        # This is only used for matching random seeding with `run_eval.py`.
        value_dict["cond_frames"] = curr_imgs + 0.0 * torch.randn_like(curr_imgs)
        value_dict["cond_aug"] = 0.0

        c2w = to_hom_pose(curr_c2ws.float())
        w2c = torch.linalg.inv(c2w)

        # camera centering
        ref_c2ws = all_c2ws
        camera_dist_2med = torch.norm(
            ref_c2ws[:, :3, 3] - ref_c2ws[:, :3, 3].median(0, keepdim=True).values,
            dim=-1,
        )
        valid_mask = camera_dist_2med <= torch.clamp(
            torch.quantile(camera_dist_2med, 0.97) * 10,
            max=1e6,
        )
        c2w[:, :3, 3] -= ref_c2ws[valid_mask, :3, 3].mean(0, keepdim=True)
        w2c = torch.linalg.inv(c2w)

        # camera normalization
        camera_dists = c2w[:, :3, 3].clone()
        translation_scaling_factor = (
            camera_scale
            if torch.isclose(
                torch.norm(camera_dists[0]), torch.zeros(1), atol=1e-6
            ).any()
            else (camera_scale / torch.norm(camera_dists[0]))
        )
        w2c[:, :3, 3] *= translation_scaling_factor
        c2w[:, :3, 3] *= translation_scaling_factor
        value_dict["plucker_coordinate"] = get_plucker_coordinates(
            extrinsics_src=w2c[0],
            extrinsics=w2c,
            intrinsics=curr_Ks.float().clone(),
            mode="plucker",
            rel_zero_translation=True,
            target_size=(H // F, W // F),
            return_grid_cam=True,
        )[0]

        value_dict["c2w"] = c2w  # used by guider
        value_dict["K"] = curr_Ks  # used by guider
        value_dict["camera_mask"] = torch.zeros(T, dtype=torch.bool)
        value_dict["camera_mask"][curr_input_camera_indices] = True

        return value_dict

    def pad_indices(
        self,
        input_indices: list[int],
        test_indices: list[int],
        padding_mode: Literal["first", "last", "none"] = "last",
    ):
        T = self.cfg.context_window
        assert padding_mode in ["last", "none"], "`first` padding is not supported yet."
        if padding_mode == "last":
            padded_indices = [
                i for i in range(T) if i not in (input_indices + test_indices)
            ]
        else:
            padded_indices = []
        input_selects = list(range(len(input_indices)))
        test_selects = list(range(len(test_indices)))
        if max(input_indices) > max(test_indices):
            # last elem from input
            input_selects += [input_selects[-1]] * len(padded_indices)
            input_indices = input_indices + padded_indices
            sorted_inds = np.argsort(np.array(input_indices))
            input_indices = [input_indices[ind] for ind in sorted_inds]
            input_selects = [input_selects[ind] for ind in sorted_inds]
        else:
            # last elem from test
            test_selects += [test_selects[-1]] * len(padded_indices)
            test_indices = test_indices + padded_indices
            sorted_inds = np.argsort(test_indices)
            test_indices = [test_indices[ind] for ind in sorted_inds]
            test_selects = [test_selects[ind] for ind in sorted_inds]

        if padding_mode == "last":
            input_maps = np.array([-1] * T)
            test_maps = np.array([-1] * T)
        else:
            input_maps = np.array([-1] * (len(input_indices) + len(test_indices)))
            test_maps = np.array([-1] * (len(input_indices) + len(test_indices)))
        input_maps[input_indices] = input_selects
        test_maps[test_indices] = test_selects
        return input_indices, test_indices, input_maps, test_maps

    def assemble(
        self,
        input,
        test,
        input_maps,
        test_maps,
    ):
        # Support padding for legacy reason, the right way is to do the
        # attention masking.
        T = len(input_maps)
        assembled = torch.zeros_like(test[-1:]).repeat_interleave(T, dim=0)
        assembled[input_maps != -1] = input[input_maps[input_maps != -1]]
        assembled[test_maps != -1] = test[test_maps[test_maps != -1]]
        assert np.logical_xor(input_maps != -1, test_maps != -1).all()
        return assembled

    def get_batch(
        self,
        keys,
        value_dict: dict,
        N: list[int],
        device: str = "cuda",
        T: int = 21,
        additional_batch_uc_fields: list[str] = [],
    ):
        # Hardcoded demo setups; might undergo some changes in the future
        batch = {}
        batch_uc = {}
        prod_N = np.prod(np.array(N)).item()

        for key in keys:
            if key == "txt":
                batch["txt"] = [value_dict["prompt"]] * prod_N
                batch_uc["txt"] = [value_dict["negative_prompt"]] * prod_N
            elif key == "cond_aug":
                batch[key] = repeat(
                    torch.tensor([value_dict["cond_aug"]]).to(device),
                    "1 -> b",
                    b=prod_N,
                )
            elif (
                key == "cond_frames"
                or key == "cond_frames_mask"
                or key == "cond_frames_without_noise"
            ):
                value = value_dict[key]
                batch[key] = repeat(
                    value if isinstance(value, torch.Tensor) else torch.tensor(value),
                    "n ... -> (b n) ...",
                    b=prod_N // T,
                ).to(device)
            elif (
                key == "polar_rad"
                or key == "azimuth_rad"
                or key == "plucker_coordinate"
                or key == "camera_mask"
            ):
                value = value_dict[key]
                batch[key] = repeat(
                    value if isinstance(value, torch.Tensor) else torch.tensor(value),
                    "n ... -> (b n) ...",
                    b=prod_N // T,
                ).to(device)
                # for logging as gt
                if key == "plucker_coordinate":
                    (
                        batch["plucker_coordinate_direction"],
                        batch["plucker_coordinate_moment"],
                    ) = batch["plucker_coordinate"].chunk(2, dim=1)
            else:
                batch[key] = value_dict[key]

        # for logging as gt
        for key in ["depth_lowres", "depth_lowres/raw", "point"]:
            if key in value_dict:
                batch[key] = repeat(
                    value_dict[key],
                    "n ... -> (b n) ...",
                    b=prod_N // T,
                ).to(device)

        if T is not None:
            batch["num_video_frames"] = T

        for key in batch.keys():
            if key not in batch_uc and isinstance(batch[key], torch.Tensor):
                batch_uc[key] = torch.clone(batch[key])
            elif key in additional_batch_uc_fields and key not in batch_uc:
                batch_uc[key] = copy.copy(batch[key])
        return batch, batch_uc

    def get_traj_fn(
        self,
        traj: Literal[
            "orbit",
            "turntable",
            "lemniscate",
            "spiral",
            "dolly zoom-in",
            "dolly zoom-out",
            "zoom-in",
            "zoom-out",
            "pan-forward",
            "pan-backward",
            "pan-up",
            "pan-down",
            "pan-left",
            "pan-right",
            "roll",
        ],
    ):
        if traj in ["orbit"]:

            def traj_fn(
                ref_c2w: torch.Tensor,
                ref_K: torch.Tensor,
                num_frames: int,
                endpoint: bool = False,
            ):
                return (
                    torch.linalg.inv(
                        get_arc_horizontal_w2cs(
                            torch.linalg.inv(ref_c2w),
                            torch.tensor([0, 0, 10]),
                            None,
                            num_frames,
                            endpoint=True,
                        )
                    )
                    if endpoint
                    else torch.linalg.inv(
                        get_arc_horizontal_w2cs(
                            torch.linalg.inv(ref_c2w),
                            torch.tensor([0, 0, 10]),
                            None,
                            num_frames + 1,
                            endpoint=False,
                        )
                    )[1:],
                    repeat(ref_K, "i j -> n i j", n=num_frames),
                    2.0,
                )

        elif traj in ["turntable"]:

            def traj_fn(
                ref_c2w: torch.Tensor,
                ref_K: torch.Tensor,
                num_frames: int,
                endpoint: bool = False,
            ):
                return (
                    torch.linalg.inv(
                        get_arc_horizontal_w2cs(
                            torch.linalg.inv(ref_c2w),
                            torch.tensor([0, 0, -10]),
                            None,
                            num_frames,
                            face_off=True,
                            endpoint=True,
                        )
                    )
                    if endpoint
                    else torch.linalg.inv(
                        get_arc_horizontal_w2cs(
                            torch.linalg.inv(ref_c2w),
                            torch.tensor([0, 0, -10]),
                            None,
                            num_frames + 1,
                            face_off=True,
                            endpoint=False,
                        )
                    )[1:],
                    repeat(ref_K, "i j -> n i j", n=num_frames),
                    0.1,
                )

        elif traj in ["lemniscate"]:

            def traj_fn(
                ref_c2w: torch.Tensor,
                ref_K: torch.Tensor,
                num_frames: int,
                endpoint: bool = False,
            ):
                return (
                    torch.linalg.inv(
                        get_lemniscate_w2cs(
                            torch.linalg.inv(ref_c2w),
                            torch.tensor([0, 0, 10]),
                            None,
                            num_frames,
                            degree=60.0,
                            endpoint=True,
                        )
                    )
                    if endpoint
                    else torch.linalg.inv(
                        get_lemniscate_w2cs(
                            torch.linalg.inv(ref_c2w),
                            torch.tensor([0, 0, 10]),
                            None,
                            num_frames + 1,
                            degree=60.0,
                            endpoint=False,
                        )
                    )[1:],
                    repeat(ref_K, "i j -> n i j", n=num_frames),
                    2.0,
                )

        elif traj in ["spiral"]:

            def traj_fn(
                ref_c2w: torch.Tensor,
                ref_K: torch.Tensor,
                num_frames: int,
                endpoint: bool = False,
            ):
                c2ws = (
                    generate_spiral_path(
                        ref_c2w[None].numpy() @ np.diagflat([1, -1, -1, 1]),
                        np.array([1, 5]),
                        n_frames=num_frames,
                        n_rots=2,
                        zrate=0.5,
                        radii=[1.0, 1.0, 0.5],
                        endpoint=True,
                    )
                    if endpoint
                    else generate_spiral_path(
                        ref_c2w[None].numpy() @ np.diagflat([1, -1, -1, 1]),
                        np.array([1, 5]),
                        n_frames=num_frames + 1,
                        n_rots=2,
                        zrate=0.5,
                        radii=[1.0, 1.0, 0.5],
                        endpoint=False,
                    )[1:]
                )
                c2ws = c2ws @ np.diagflat([1, -1, -1, 1])
                c2ws = to_hom_pose(torch.as_tensor(c2ws).float())
                # The original spiral path does not pass through the reference.
                # So we do the relative here.
                c2ws = ref_c2w @ torch.linalg.inv(c2ws[:1]) @ c2ws
                return c2ws, repeat(ref_K, "i j -> n i j", n=num_frames), 2.0

        elif traj in [
            "dolly zoom-in",
            "dolly zoom-out",
            "zoom-in",
            "zoom-out",
        ]:

            def traj_fn(
                ref_c2w: torch.Tensor,
                ref_K: torch.Tensor,
                num_frames: int,
                endpoint: bool = False,
            ):
                if traj.startswith("dolly"):
                    direction = "backward" if traj == "dolly zoom-in" else "forward"
                    c2ws = (
                        torch.linalg.inv(
                            get_panning_w2cs(
                                torch.linalg.inv(ref_c2w),
                                torch.tensor([0, 0, 100]),
                                None,
                                num_frames,
                                endpoint=True,
                                direction=direction,
                            )
                        )
                        if endpoint
                        else torch.linalg.inv(
                            get_panning_w2cs(
                                torch.linalg.inv(ref_c2w),
                                torch.tensor([0, 0, 100]),
                                None,
                                num_frames + 1,
                                endpoint=False,
                                direction=direction,
                            )
                        )[1:]
                    )
                else:
                    c2ws = repeat(ref_c2w, "i j -> n i j", n=num_frames)
                # TODO(hangg): Here always assume DEFAULT_FOV_RAD, need to
                # improve to support general case.
                fov_rad_start = DEFAULT_FOV_RAD
                fov_rad_end = (
                    0.28 if traj.endswith("zoom-in") else 1.5
                ) * DEFAULT_FOV_RAD
                return (
                    c2ws,
                    torch.cat(
                        [
                            get_default_intrinsics(
                                float(
                                    fov_rad_start
                                    + ratio * (fov_rad_end - fov_rad_start)
                                )
                            )
                            for ratio in torch.linspace(
                                0,
                                1,
                                num_frames + 1 - endpoint,
                            )
                        ],
                        dim=0,
                    )[1 - endpoint :],
                    10.0,
                )

        elif traj in [
            "pan-forward",
            "pan-backward",
            "pan-up",
            "pan-down",
            "pan-left",
            "pan-right",
        ]:

            def traj_fn(
                ref_c2w: torch.Tensor,
                ref_K: torch.Tensor,
                num_frames: int,
                endpoint: bool = False,
            ):
                return (
                    torch.linalg.inv(
                        get_panning_w2cs(
                            torch.eye(4),
                            torch.tensor([0, 0, 100]),
                            None,
                            num_frames,
                            endpoint=True,
                            direction=traj.removeprefix("pan-"),
                        )
                    )
                    if endpoint
                    else torch.linalg.inv(
                        get_panning_w2cs(
                            torch.eye(4),
                            torch.tensor([0, 0, 100]),
                            None,
                            num_frames + 1,
                            endpoint=True,
                            direction=traj.removeprefix("pan-"),
                        )
                    )[1:],
                    repeat(ref_K, "i j -> n i j", n=num_frames),
                    10.0,
                )

        elif "roll" in traj:

            def traj_fn(
                ref_c2w: torch.Tensor,
                ref_K: torch.Tensor,
                num_frames: int,
                endpoint: bool = False,
            ):
                return (
                    torch.linalg.inv(
                        get_roll_w2cs(
                            torch.eye(4),
                            torch.tensor([0, 0, 10]),
                            None,
                            num_frames,
                            endpoint=True,
                        )
                    )
                    if endpoint
                    else torch.linalg.inv(
                        get_roll_w2cs(
                            torch.eye(4),
                            torch.tensor([0, 0, 10]),
                            None,
                            num_frames + 1,
                            endpoint=True,
                        )
                    )[1:],
                    repeat(ref_K, "i j -> n i j", n=num_frames),
                    2.0,
                )

        else:
            raise ValueError(f"Unsupported trajectory: {traj}")

        return traj_fn

    def prepare(
        self,
        img: np.ndarray,
        traj: Literal[
            "orbit",
            "turntable",
            "lemniscate",
            "spiral",
            "dolly zoom-in",
            "dolly zoom-out",
            "zoom-in",
            "zoom-out",
            "pan-forward",
            "pan-backward",
            "pan-up",
            "pan-down",
            "pan-left",
            "pan-right",
            "roll",
        ],
        num_targets: int = 80,
        shorter: int = 576,
        keep_aspect: bool = True,
    ):
        traj_fn = self.get_traj_fn(traj)

        """
        # Has to be 64 multiple for the network.
        shorter = round(shorter / 64) * 64
        for img, K in zip(input_imgs, input_Ks):
            img = rearrange(img, "h w c -> 1 c h w")
            if not keep_aspect:
                img, K = transform_img_and_K(img, (shorter, shorter), K=K[None])
            else:
                img, K = transform_img_and_K(img, shorter, K=K[None], size_stride=64)
            K = K / np.array([img.shape[-1], img.shape[-2], 1])[:, None]
            new_input_imgs.append(img)
            new_input_Ks.append(K)
        input_imgs = torch.cat(new_input_imgs, 0)
        input_imgs = rearrange(input_imgs, "b c h w -> b h w c")[..., :3]
        input_Ks = torch.cat(new_input_Ks, 0)
        """

        num_inputs = 1
        # Has to be 64 multiple for the network.
        shorter = round(shorter / 64) * 64
        input_imgs = repeat(
            torch.as_tensor(img / 255.0, dtype=torch.float32), "h w c -> n c h w", n=num_inputs
        )
        input_Ks = repeat(get_default_intrinsics(), "1 i j -> n i j", n=num_inputs)
        if not keep_aspect:
            input_imgs, K = transform_img_and_K(input_imgs, (shorter, shorter), K=input_Ks)
        else:
            input_imgs, K = transform_img_and_K(input_imgs, shorter, K=input_Ks, size_stride=64)

        
        #input_imgs = transform_img_and_K(input_imgs, None, self.cfg.target_wh)[0]
        input_c2ws = repeat(torch.eye(4), "i j -> n i j", n=num_inputs)
        target_c2ws, target_Ks, camera_scale = traj_fn(
            input_c2ws[0], input_Ks[0], num_targets
        )
        if self.cfg.chunk_strategy.startswith("interp"):
            num_anchors = max(
                np.ceil(
                    num_targets
                    / (
                        self.cfg.context_window
                        - 2
                        - (num_inputs if self.cfg.chunk_strategy == "interp-gt" else 0)
                    )
                )
                .astype(int)
                .item()
                + 1,
                self.cfg.context_window - num_inputs,
            )
            include_start_end = True
        else:
            num_anchors = max(self.cfg.context_window - num_inputs, 0)
            include_start_end = False
        anchor_c2ws, anchor_Ks, _ = traj_fn(
            input_c2ws[0], input_Ks[0], num_anchors, endpoint=include_start_end
        )

        # Get indices.
        input_indices = list(range(num_inputs))
        anchor_indices = np.linspace(
            1,
            num_targets,
            num_anchors + 1 - include_start_end,
            endpoint=include_start_end,
        )[1 - include_start_end :].tolist()
        target_indices = np.arange(num_inputs, num_inputs + num_targets).tolist()

        return StableViewsSingleImageData(
            input_imgs,
            input_c2ws,
            input_Ks,
            target_c2ws,
            target_Ks,
            anchor_c2ws,
            anchor_Ks,
            num_inputs,
            num_targets,
            num_anchors,
            input_indices,
            anchor_indices,
            target_indices,
            camera_scale,
        )

    @torch.inference_mode()
    def render_video(
        self,
        img: np.ndarray,
        traj: Literal[
            "orbit",
            "turntable",
            "lemniscate",
            "spiral",
            "dolly zoom-in",
            "dolly zoom-out",
            "zoom-in",
            "zoom-out",
            "pan-forward",
            "pan-backward",
            "pan-up",
            "pan-down",
            "pan-left",
            "pan-right",
            "roll",
        ],
        num_targets: int,
        seed: int = 23,
        shorter: int = 576,
    ):
        keep_aspect = False
        data = self.prepare(img, traj, num_targets, shorter, keep_aspect)
        data_name = hashlib.sha256(img.tobytes()).hexdigest()[:16]

        output_dir = osp.join(self.cfg.output_root, data_name, f"{traj}_{num_targets}_{shorter}_{seed}")

        # If the input image is one of the examples (verified by hash) and cached videos exist, return them immediately.
        if data_name in EXAMPLE_HASHES:
            first_pass_path = osp.join(output_dir, "first_pass.mp4")
            second_pass_path = osp.join(output_dir, "second_pass.mp4")
            if os.path.exists(first_pass_path):
                if os.path.exists(second_pass_path):
                    yield first_pass_path, second_pass_path
                    return

        #samplers = init_sampling_no_st(options=self.cfg.options)
        samplers = create_samplers(
            self.cfg.options["guider_types"],
            DISCRETIZATION,
            [self.cfg.context_window, self.cfg.context_window],
            self.cfg.options["num_steps"],
        )

        seed_everything(seed)

        (
            _,
            input_inds_per_chunk,
            input_sels_per_chunk,
            anchor_inds_per_chunk,
            anchor_sels_per_chunk,
        ) = chunk_input_and_test(
            self.cfg.context_window,
            data.input_c2ws,
            data.anchor_c2ws,
            data.input_indices,
            data.anchor_indices,
            task="img2trajvid",
            chunk_strategy=self.cfg.chunk_strategy_first_pass,
            gt_input_inds=list(range(data.num_inputs)),
            options=self.cfg.options,
        )
        print(
            f"Two passes (first) - chunking with `{self.cfg.chunk_strategy_first_pass}` strategy: total "
            f"{len(input_inds_per_chunk)} forward(s) ..."
        )

        gradio_pbar = gr.Progress(track_tqdm=True)

        all_samples = {}
        all_anchor_inds = []
        for i, (
            chunk_input_inds,
            chunk_input_sels,
            chunk_anchor_inds,
            chunk_anchor_sels,
        ) in tqdm(
            enumerate(
                zip(
                    input_inds_per_chunk,
                    input_sels_per_chunk,
                    anchor_inds_per_chunk,
                    anchor_sels_per_chunk,
                )
            ),
            total=len(input_inds_per_chunk),
            leave=False,
        ):
            (
                curr_input_sels,
                curr_anchor_sels,
                curr_input_maps,
                curr_anchor_maps,
            ) = self.pad_indices(
                chunk_input_sels,
                chunk_anchor_sels,
                padding_mode="last",
            )
            curr_imgs, curr_imgs_clip, curr_c2ws, curr_Ks, curr_depths = [
                self.assemble(
                    input=x[chunk_input_inds],
                    test=y[chunk_anchor_inds],
                    input_maps=curr_input_maps,
                    test_maps=curr_anchor_maps,
                )
                for x, y in zip(
                    [
                        torch.cat(
                            [
                                data.input_imgs * 2.0 - 1.0,
                                get_k_from_dict(all_samples, "samples-rgb").to(
                                    data.input_imgs.device
                                ),
                            ],
                            dim=0,
                        ),
                        torch.cat(
                            [
                                data.input_imgs * 2.0 - 1.0,
                                get_k_from_dict(all_samples, "samples-rgb").to(
                                    data.input_imgs.device
                                ),
                            ],
                            dim=0,
                        ),
                        torch.cat(
                            [data.input_c2ws, data.anchor_c2ws[all_anchor_inds]], dim=0
                        ),
                        torch.cat(
                            [data.input_Ks, data.anchor_Ks[all_anchor_inds]], dim=0
                        ),
                        torch.cat(
                            [data.input_Ks, data.anchor_Ks[all_anchor_inds]], dim=0
                        ),
                    ],  # procedurally append generated prior views to the input views
                    [
                        repeat(
                            torch.zeros_like(data.input_imgs[:1]),
                            "1 c h w -> n c h w",
                            n=data.num_anchors,
                        ),
                        repeat(
                            torch.zeros_like(data.input_imgs[:1]),
                            "1 c h w -> n c h w",
                            n=data.num_anchors,
                        ),
                        data.anchor_c2ws,
                        data.anchor_Ks,
                        data.anchor_Ks,
                    ],
                )
            ]
            value_dict = self.get_value_dict(
                curr_imgs,
                curr_imgs_clip,
                curr_input_sels,
                curr_c2ws,
                curr_Ks,
                list(range(self.cfg.context_window)),
                curr_depths,
                all_c2ws=torch.cat([data.input_c2ws, data.target_c2ws], 0),
                as_shuffled=(
                    self.cfg.options["as_shuffled"]
                    and any(
                        map(
                            lambda x: x in self.cfg.chunk_strategy_first_pass,
                            ["nearest", "gt"],
                        )
                    )  # "interp" not in chunk_strategy_first_pass
                ),
                camera_scale=data.camera_scale,
            )

            samples = do_sample(
                MODEL,
                AE,
                CONDITIONER,
                DENOISER,
                (
                    samplers[1]
                    if len(samplers) > 1
                    and self.cfg.options.get("ltr_first_pass", False)
                    and self.cfg.chunk_strategy_first_pass != "gt"
                    and i > 0
                    else samplers[0]
                ),
                value_dict,
                self.cfg.target_wh[1],
                self.cfg.target_wh[0],
                4,
                8,
                T=self.cfg.context_window,
                cfg=self.cfg.options["cfg"][0],
                gradio_pbar=gradio_pbar,
                **{k: self.cfg.options[k] for k in self.cfg.options if k not in ["cfg", "T"]},
            )

            samples = decode_output(samples, self.cfg.context_window, chunk_anchor_sels)
            extend_dict(all_samples, samples)
            all_anchor_inds.extend(chunk_anchor_inds)

        first_pass_samples = rearrange(
            (all_samples["samples-rgb/image"] / 2.0 + 0.5).clamp(0.0, 1.0).cpu().numpy()
            * 255.0,
            "n c h w -> n h w c",
        ).astype(np.uint8)

        if data_name not in EXAMPLE_HASHES:
            first_pass_dir = osp.join(output_dir, "first_pass")
            os.makedirs(first_pass_dir, exist_ok=True)
            first_pass_path = osp.join(output_dir, "first_pass.mp4")
            iio.imwrite(first_pass_path, first_pass_samples, fps=5.0)
            yield first_pass_path, None
        else:
            first_pass_tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            iio.imwrite(first_pass_tmp_file.name, first_pass_samples, fps=5.0)
            yield first_pass_tmp_file.name, None

        assert (
            data.anchor_indices is not None
        ), "`anchor_frame_indices` needs to be set if using 2-pass sampling."
        anchor_argsort = np.argsort(
            np.array(data.input_indices + data.anchor_indices)
        ).tolist()
        anchor_indices = np.array(data.input_indices + data.anchor_indices)[
            anchor_argsort
        ].tolist()
        gt_input_inds = [
            anchor_argsort.index(i) for i in range(data.input_c2ws.shape[0])
        ]

        anchor_imgs = torch.cat(
            [data.input_imgs, get_k_from_dict(all_samples, "samples-rgb") / 2.0 + 0.5],
            dim=0,
        )[anchor_argsort]
        anchor_c2ws = torch.cat([data.input_c2ws, data.anchor_c2ws], dim=0)[
            anchor_argsort
        ]
        anchor_Ks = torch.cat([data.input_Ks, data.anchor_Ks], dim=0)[anchor_argsort]

        update_kv_for_dict(all_samples, "samples-rgb", anchor_imgs)
        update_kv_for_dict(all_samples, "samples-c2ws", anchor_c2ws)
        update_kv_for_dict(all_samples, "samples-intrinsics", anchor_Ks)

        (
            _,
            anchor_inds_per_chunk,
            anchor_sels_per_chunk,
            target_inds_per_chunk,
            target_sels_per_chunk,
        ) = chunk_input_and_test(
            self.cfg.context_window,
            anchor_c2ws,
            data.target_c2ws,
            anchor_indices,
            data.target_indices,
            task="img2trajvid",
            chunk_strategy=self.cfg.chunk_strategy,
            gt_input_inds=gt_input_inds,
            options=self.cfg.options,
        )
        print(
            f"Two passes (second) - chunking with `{self.cfg.chunk_strategy}` strategy: total "
            f"{len(anchor_inds_per_chunk)} forward(s) ..."
        )

        all_samples = {}
        all_target_inds = []
        for i, (
            chunk_anchor_inds,
            chunk_anchor_sels,
            chunk_target_inds,
            chunk_target_sels,
        ) in tqdm(
            enumerate(
                zip(
                    anchor_inds_per_chunk,
                    anchor_sels_per_chunk,
                    target_inds_per_chunk,
                    target_sels_per_chunk,
                )
            ),
            total=len(anchor_inds_per_chunk),
            leave=False,
        ):
            (
                curr_anchor_sels,
                curr_target_sels,
                curr_anchor_maps,
                curr_target_maps,
            ) = self.pad_indices(
                chunk_anchor_sels,
                chunk_target_sels,
                padding_mode="last",
            )
            curr_imgs, curr_imgs_clip, curr_c2ws, curr_Ks, curr_depths = [
                self.assemble(
                    input=x[chunk_anchor_inds],
                    test=y[chunk_target_inds],
                    input_maps=curr_anchor_maps,
                    test_maps=curr_target_maps,
                )
                for x, y in zip(
                    [
                        anchor_imgs * 2.0 - 1.0,
                        anchor_imgs * 2.0 - 1.0,
                        anchor_c2ws,
                        anchor_Ks,
                        anchor_Ks,
                    ],
                    [
                        repeat(
                            torch.zeros_like(data.input_imgs[:1]),
                            "1 c h w -> n c h w",
                            n=num_targets,
                        ),
                        repeat(
                            torch.zeros_like(data.input_imgs[:1]),
                            "1 c h w -> n c h w",
                            n=num_targets,
                        ),
                        data.target_c2ws,
                        data.target_Ks,
                        data.target_Ks,
                    ],
                )
            ]
            value_dict = self.get_value_dict(
                curr_imgs,
                curr_imgs_clip,
                curr_anchor_sels,
                curr_c2ws,
                curr_Ks,
                list(range(self.cfg.context_window)),
                curr_depths,
                all_c2ws=torch.cat([data.input_c2ws, data.target_c2ws], 0),
                as_shuffled=(
                    self.cfg.options["as_shuffled"]
                    and any(
                        map(
                            lambda x: x in self.cfg.chunk_strategy,
                            ["nearest", "gt"],
                        )
                    )  # "interp" not in chunk_strategy
                ),
                camera_scale=data.camera_scale,
            )
            samples = do_sample(
                MODEL,
                AE,
                CONDITIONER,
                DENOISER,
                samplers[1] if len(samplers) > 1 else samplers[0],
                value_dict,
                self.cfg.target_wh[1],
                self.cfg.target_wh[0],
                4,
                8,
                T=self.cfg.context_window,
                cfg=self.cfg.options["cfg"][1],
                gradio_pbar=gradio_pbar,
                **{k: self.cfg.options[k] for k in self.cfg.options if k not in ["cfg", "T"]},
            )
            samples = decode_output(samples, self.cfg.context_window, chunk_target_sels)
            extend_dict(all_samples, samples)
            all_target_inds.extend(chunk_target_inds)

        all_samples = {
            key: value[np.argsort(all_target_inds)]
            for key, value in all_samples.items()
        }
        second_pass_samples = rearrange(
            (all_samples["samples-rgb/image"] / 2.0 + 0.5).clamp(0.0, 1.0).cpu().numpy()
            * 255.0,
            "n c h w -> n h w c",
        ).astype(np.uint8)
        if data_name not in EXAMPLE_HASHES:
            second_pass_dir = osp.join(output_dir, "second_pass")
            os.makedirs(second_pass_dir, exist_ok=True)
            second_pass_path = osp.join(output_dir, "second_pass.mp4")
            iio.imwrite(second_pass_path, second_pass_samples, fps=30.0)
            yield first_pass_path, second_pass_path
        else:
            second_pass_tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            iio.imwrite(second_pass_tmp_file.name, second_pass_samples, fps=30.0)
            yield first_pass_tmp_file.name, second_pass_tmp_file.name


class StableViewsRenderer(object):
    def __init__(self, server: viser.ViserServer):
        self.server = server
        self.gui_state = None

    def preprocess(
        self,
        input_img_tuples: list[tuple[str, None]],
        shorter: int,
        keep_aspect: bool,
    ) -> tuple[dict, dict, dict]:
        img_paths = [p for (p, _) in input_img_tuples]
        (
            input_imgs,
            input_Ks,
            input_c2ws,
            points,
            point_colors,
        ) = DUST3R.infer_cameras_and_points(img_paths)
        num_inputs = len(img_paths)
        if num_inputs == 1:
            input_imgs, input_Ks, input_c2ws, points, point_colors = (
                input_imgs[:1],
                input_Ks[:1],
                input_c2ws[:1],
                points[:1],
                point_colors[:1],
            )
        # Normalize the scene.
        point_chunks = [p.shape[0] for p in points]
        point_indices = np.cumsum(point_chunks)[:-1]
        input_c2ws, points, _ = normalize_scene(  # type: ignore
            input_c2ws,
            np.concatenate(points, 0),
            camera_center_method="poses",
        )
        points = np.split(points, point_indices, 0)
        # Scale camera and points for viewport visualization.
        scene_scale = np.concatenate([input_c2ws[:, :3, 3], *points], 0).ptp(-1).mean()
        input_c2ws[:, :3, 3] /= scene_scale
        points = [point / scene_scale for point in points]
        input_imgs = [
            torch.as_tensor(img / 255.0, dtype=torch.float32) for img in input_imgs
        ]
        input_Ks = torch.as_tensor(input_Ks)
        input_c2ws = torch.as_tensor(input_c2ws)
        new_input_imgs, new_input_Ks = [], []
        # Has to be 64 multiple for the network.
        shorter = round(shorter / 64) * 64
        for img, K in zip(input_imgs, input_Ks):
            img = rearrange(img, "h w c -> 1 c h w")
            if not keep_aspect:
                img, K = transform_img_and_K(img, (shorter, shorter), K=K[None])
            else:
                img, K = transform_img_and_K(img, shorter, K=K[None], size_stride=64)
            K = K / np.array([img.shape[-1], img.shape[-2], 1])[:, None]
            new_input_imgs.append(img)
            new_input_Ks.append(K)
        input_imgs = torch.cat(new_input_imgs, 0)
        input_imgs = rearrange(input_imgs, "b c h w -> b h w c")[..., :3]
        input_Ks = torch.cat(new_input_Ks, 0)
        return (
            {
                "input_imgs": input_imgs,
                "input_Ks": input_Ks,
                "input_c2ws": input_c2ws,
                "input_wh": (input_imgs.shape[2], input_imgs.shape[1]),
                "points": points,
                "point_colors": point_colors,
                "scene_scale": scene_scale,
            },
            gr.update(visible=False),
            gr.update(
                choices=["interp-gt", "interp"] if num_inputs <= 10 else ["interp"]
            ),
        )

    def visualize_scene(self, preprocessed: dict):
        server = self.server
        server.scene.reset()
        server.gui.reset()
        set_bkgd_color(server)

        (
            input_imgs,
            input_Ks,
            input_c2ws,
            input_wh,
            points,
            point_colors,
            scene_scale,
        ) = (
            preprocessed["input_imgs"],
            preprocessed["input_Ks"],
            preprocessed["input_c2ws"],
            preprocessed["input_wh"],
            preprocessed["points"],
            preprocessed["point_colors"],
            preprocessed["scene_scale"],
        )
        W, H = input_wh

        server.scene.set_up_direction(-input_c2ws[..., :3, 1].mean(0).numpy())

        # TODO(hangg): Sometime duster will give fx != fy, it will break the
        # viser fov logic.
        # Use first image as default fov.
        assert input_imgs[0].shape[:2] == (H, W)
        if H > W:
            init_fov = 2 * np.arctan(1 / (2 * input_Ks[0, 0, 0].item()))
        else:
            init_fov = 2 * np.arctan(1 / (2 * input_Ks[0, 1, 1].item()))
        init_fov_deg = float(init_fov / np.pi * 180.0)

        frustrums = []
        input_camera_node_prefix = "/scene_assets/cameras/"
        input_camera_node = server.scene.add_frame(
            input_camera_node_prefix, show_axes=False
        )
        for i in range(len(input_imgs)):
            K = input_Ks[i]
            frustum = server.scene.add_camera_frustum(
                f"/scene_assets/cameras/{i}",
                fov=2 * np.arctan(1 / (2 * K[1, 1].item())),
                aspect=W / H,
                scale=0.1 * scene_scale,
                image=(input_imgs[i].numpy() * 255.0).astype(np.uint8),
                wxyz=vt.SO3.from_matrix(input_c2ws[i, :3, :3].numpy()).wxyz,
                position=input_c2ws[i, :3, 3].numpy(),
            )

            def get_handler(frustum):
                def handler(event: viser.GuiEvent) -> None:
                    assert event.client_id is not None
                    client = server.get_clients()[event.client_id]
                    with client.atomic():
                        client.camera.position = frustum.position
                        client.camera.wxyz = frustum.wxyz
                        # Set look_at as the projected origin onto the
                        # frustum's forward direction.
                        look_direction = vt.SO3(frustum.wxyz).as_matrix()[:, 2]
                        position_origin = -frustum.position
                        client.camera.look_at = (
                            frustum.position
                            + np.dot(look_direction, position_origin)
                            / np.linalg.norm(position_origin)
                            * look_direction
                        )

                return handler

            frustum.on_click(get_handler(frustum))  # type: ignore
            frustrums.append(frustum)

            server.scene.add_point_cloud(
                f"/scene_assets/points/{i}",
                points[i],
                point_colors[i],
                point_size=0.01 * scene_scale,
                point_shape="circle",
            )

        self.gui_state = define_gui(
            server,
            init_fov=init_fov_deg,
            img_wh=input_wh,
            input_camera_node_list=[input_camera_node],
            scene_scale=scene_scale,
        )

    def render(
        self,
        preprocessed: dict,
        seed: int,
        chunk_strategy: str,
        cfg: float,
        camera_scale: float,
    ):
        render_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        render_dir = osp.join(WORK_DIR, render_name)

        input_imgs, input_Ks, input_c2ws, input_wh = (
            preprocessed["input_imgs"],
            preprocessed["input_Ks"],
            preprocessed["input_c2ws"],
            preprocessed["input_wh"],
        )
        W, H = input_wh
        gui_state = self.gui_state
        assert gui_state is not None and gui_state.camera_traj_list is not None
        target_c2ws, target_Ks = [], []
        for item in gui_state.camera_traj_list:
            target_c2ws.append(item["w2c"])
            assert item["img_wh"] == input_wh
            K = np.array(item["K"]).reshape(3, 3) / np.array([W, H, 1])[:, None]
            target_Ks.append(K)
        target_c2ws = torch.as_tensor(
            np.linalg.inv(np.array(target_c2ws).reshape(-1, 4, 4))
        )
        target_Ks = torch.as_tensor(np.array(target_Ks).reshape(-1, 3, 3))
        num_inputs = len(input_imgs)
        num_targets = len(target_c2ws)
        input_indices = list(range(num_inputs))
        target_indices = np.arange(num_inputs, num_inputs + num_targets).tolist()
        # Get anchor cameras.
        T = VERSION_DICT["T"]
        version_dict = copy.deepcopy(VERSION_DICT)
        num_anchors, include_start_end = infer_prior_stats(
            T,
            num_inputs,
            num_total_frames=num_targets,
            version_dict=version_dict,
        )
        # infer_prior_stats modifies T in-place.
        T = version_dict["T"]
        assert isinstance(num_anchors, int)
        anchor_indices = np.linspace(
            num_inputs,
            num_inputs + num_targets - 1,
            num_anchors + 1 - include_start_end,
            endpoint=include_start_end,
        )[1 - include_start_end :].tolist()
        anchor_c2ws = target_c2ws[
            np.linspace(0, num_targets - 1, num_anchors)
            .round()
            .astype(np.int64)
            .tolist()
        ]
        anchor_Ks = None
        # Create image conditioning.
        all_imgs_np = (
            F.pad(input_imgs, (0, 0, 0, 0, 0, 0, 0, num_targets), value=0.0).numpy()
            * 255.0
        ).astype(np.uint8)
        image_cond = {
            "img": all_imgs_np,
            "input_indices": input_indices,
            "prior_indices": anchor_indices,
            "target_indices": target_indices,
        }
        # Create camera conditioning (K is unnormalized).
        Ks_ori = torch.cat([input_Ks, target_Ks], 0)
        Ks_ori = Ks_ori / Ks_ori.new_tensor([W, H, 1])[:, None]
        camera_cond = {
            "c2w": torch.cat([input_c2ws, target_c2ws], 0),
            "K": Ks_ori,
            "input_indices": list(range(num_inputs + num_targets)),
        }
        # Run rendering.
        num_steps = 50
        options_ori = VERSION_DICT["options"]
        options = copy.deepcopy(options_ori)
        options["chunk_strategy"] = chunk_strategy
        options["video_save_fps"] = 30.0
        options["beta_linear_start"] = 5e-6
        options["log_snr_shift"] = 2.4
        options["guider_types"] = [1, 2]
        options["cfg"] = [float(cfg), 2.0]
        options["use_traj_prior"] = True
        options["camera_scale"] = camera_scale
        options["num_steps"] = num_steps
        options["encoding_t"] = 1
        options["decoding_t"] = 1
        task = "img2trajvid"
        # Get number of first pass chunks.
        T_first_pass = T[0] if isinstance(T, (list, tuple)) else T
        chunk_strategy_first_pass = options.get(
            "chunk_strategy_first_pass", "gt-nearest"
        )
        num_chunks_0 = len(
            chunk_input_and_test(
                T_first_pass,
                input_c2ws,
                anchor_c2ws,
                input_indices,
                image_cond["prior_indices"],
                options=options,
                task=task,
                chunk_strategy=chunk_strategy_first_pass,
                gt_input_inds=list(range(input_c2ws.shape[0])),
            )[1]
        )
        # Get number of second pass chunks.
        prior_indices = anchor_indices.copy()
        prior_argsort = np.argsort(input_indices + prior_indices).tolist()
        prior_indices = np.array(input_indices + prior_indices)[prior_argsort].tolist()
        gt_input_inds = [prior_argsort.index(i) for i in range(input_c2ws.shape[0])]
        traj_prior_c2ws = torch.cat([input_c2ws, anchor_c2ws], dim=0)[prior_argsort]
        T_second_pass = T[1] if isinstance(T, (list, tuple)) else T
        chunk_strategy = options.get("chunk_strategy", "nearest")
        num_chunks_1 = len(
            chunk_input_and_test(
                T_second_pass,
                traj_prior_c2ws,
                target_c2ws,
                prior_indices,
                target_indices,
                options=options,
                task=task,
                chunk_strategy=chunk_strategy,
                gt_input_inds=gt_input_inds,
            )[1]
        )

        pbar = gr.Progress(track_tqdm=True)
        video_path_generator = run_one_scene(
            task=task,
            version_dict={
                "H": H,
                "W": W,
                "T": T,
                "C": VERSION_DICT["C"],
                "f": VERSION_DICT["f"],
                "options": options,
            },
            model=MODEL,
            ae=AE,
            conditioner=CONDITIONER,
            denoiser=DENOISER,
            image_cond=image_cond,
            camera_cond=camera_cond,
            save_path=render_dir,
            use_traj_prior=True,
            traj_prior_c2ws=anchor_c2ws,
            traj_prior_Ks=anchor_Ks,
            seed=seed,
            gradio=True,
            gradio_pbar=pbar,
        )
        for i, video_path in enumerate(video_path_generator):
            if i == 0:
                yield video_path, gr.update()
            elif i == 1:
                yield video_path, gr.update(visible=False)
            else:
                gr.Error("More than two passes during rendering.")


# This is basically a copy of the original `networking.setup_tunnel` function,
# but it also returns the tunnel object for proper cleanup.
def setup_tunnel(
    local_host: str, local_port: int, share_token: str, share_server_address: str | None
) -> tuple[str, Tunnel]:
    share_server_address = (
        networking.GRADIO_SHARE_SERVER_ADDRESS
        if share_server_address is None
        else share_server_address
    )
    if share_server_address is None:
        try:
            response = httpx.get(networking.GRADIO_API_SERVER, timeout=30)
            payload = response.json()[0]
            remote_host, remote_port = payload["host"], int(payload["port"])
            certificate = payload["root_ca"]
            Path(CERTIFICATE_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(CERTIFICATE_PATH, "w") as f:
                f.write(certificate)
        except Exception as e:
            raise RuntimeError(
                "Could not get share link from Gradio API Server."
            ) from e
    else:
        remote_host, remote_port = share_server_address.split(":")
        remote_port = int(remote_port)
    tunnel = Tunnel(remote_host, remote_port, local_host, local_port, share_token)
    address = tunnel.start_tunnel()
    return address, tunnel


def set_bkgd_color(server: viser.ViserServer | viser.ClientHandle):
    server.scene.set_background_image(np.array([[[39, 39, 42]]], dtype=np.uint8))


def start_server(request: gr.Request):
    if len(SERVERS) >= MAX_SESSIONS:
        raise gr.Error(
            f"Maximum session count reached. Please try again later. "
            "You can also try running our demo locally."
        )
    server = viser.ViserServer()

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.gui.configure_theme(
            dark_mode=True,
            show_share_button=False,
            control_layout="collapsible",
        )
        set_bkgd_color(client)

    print(f"Starting server {server.get_port()}")
    server_url, tunnel = setup_tunnel(
        local_host=server.get_host(),
        local_port=server.get_port(),
        share_token=secrets.token_urlsafe(32),
        share_server_address=None,
    )
    SERVERS[request.session_hash] = (server, tunnel)
    if server_url is None:
        raise gr.Error(
            "Failed to get a viewport URL. Please check your network connection."
        )
    # Give it enough time to start.
    time.sleep(1)
    return StableViewsRenderer(server), gr.HTML(
        f'<iframe src="{server_url}" style="display: block; margin: auto; width: 100%; height: 60vh; overflow: scroll;" frameborder="0"></iframe>',
        container=True,
    )


def stop_server(request: gr.Request):
    if request.session_hash in SERVERS:
        server, tunnel = SERVERS.pop(request.session_hash)
        print(f"Stopping server {server.get_port()}")
        server.stop()
        tunnel.kill()


def get_examples(selection: gr.SelectData):
    index = selection.index
    return (
        gr.Gallery(EXAMPLE_MAP[index][1], visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.Gallery(visible=False),
    )

def main(server_port: int | None = None, share: bool = True):
    with gr.Blocks() as demo:
        # Assign the Tabs container to a variable so that we can attach a change event.
        tabs = gr.Tabs()
        with tabs:
            with gr.Tab("Basic (Single Image)"):
                single_image_renderer = StableViewsSingleImageRenderer(StableViewsSingleImageConfig())
                with gr.Row(variant="panel"):
                    with gr.Row():
                        gr.Markdown(
                            """
                        # Stable Views Single Image gradio demo

                        ## Workflow

                        1. Upload an image.
                        2. Choose camera trajecotry and #frames, click "Render".
                        3. Three videos will be generated: preprocessed input, intermediate output, and final output.

                        > For a 80-frame video, intermediate output takes 20s, final output takes ~5 minutes.
                        > Our model currently doesn't work well with human and animal images.
                                                """
                        )
                    with gr.Row():
                        with gr.Column():
                            uploaded_img = gr.Image(
                                type="numpy", label="Upload", height=single_image_renderer.cfg.target_wh[1]
                            )
                            gr.Examples(
                                examples=sorted(glob(f"{EXAMPLE_DIR}*png")),
                                inputs=[uploaded_img],
                                label="Examples",
                            )
                            traj_handle = gr.Dropdown(
                                choices=[
                                    "orbit",
                                    "turntable",
                                    "lemniscate",
                                    "spiral",
                                    "dolly zoom-in",
                                    "dolly zoom-out",
                                    "zoom-in",
                                    "zoom-out",
                                    "pan-forward",
                                    "pan-backward",
                                    "pan-up",
                                    "pan-down",
                                    "pan-left",
                                    "pan-right",
                                    "roll",
                                ],
                                label="Preset trajectory",
                            )
                            num_targets_handle = gr.Slider(30, 150, 80, label="#Frames")
                            seed_handle = gr.Number(value=single_image_renderer.cfg.seed, label="Random seed")
                            shorter = gr.Number(
                                value=576, label="Resize", scale=3
                            )
                            render_btn = gr.Button("Render")
                        with gr.Column():
                            fast_video = gr.Video(
                                label="Intermediate output [1/2]", autoplay=True, loop=True
                            )
                            slow_video = gr.Video(
                                label="Final output [2/2]", autoplay=True, loop=True
                            )
                            render_btn.click(
                                single_image_renderer.render_video,
                                inputs=[uploaded_img, traj_handle, num_targets_handle, seed_handle, shorter],
                                outputs=[fast_video, slow_video],
                            )

            with gr.Tab("Advanced"):
                renderer = gr.State()
                render_btn = gr.Button("Render video", interactive=False, render=False)
                gr.Timer(0.1).tick(
                    lambda renderer: gr.update(
                        interactive=renderer is not None
                        and renderer.gui_state is not None
                        and renderer.gui_state.camera_traj_list is not None
                    ),
                    inputs=[renderer],
                    outputs=[render_btn],
                )
                with gr.Row():
                    gr.Markdown("**The pointcloud shown below is not used as an input to the model. It's for visualization purposes only.**")
                with gr.Row():
                    viewport = gr.HTML(container=True)
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            preprocess_btn = gr.Button("Preprocess images")
                            preprocess_progress = gr.Textbox(
                                label="",
                                visible=False,
                                interactive=False,
                            )
                        with gr.Group():
                            input_imgs = gr.Gallery(interactive=True, label="Input")
                            # Define example images (gradio doesn't support variable length
                            # examples so we need to hack it).
                            example_imgs = gr.Gallery(
                                [e[0] for e in EXAMPLE_MAP],
                                allow_preview=False,
                                preview=False,
                                label="Example",
                                columns=20,
                                rows=1,
                                height=115,
                            )
                            example_imgs_expander = gr.Gallery(
                                visible=False,
                                interactive=False,
                                label="Example",
                                preview=True,
                                columns=20,
                                rows=1,
                            )
                            chunk_strategy = gr.Dropdown(
                                ["interp-gt", "interp"],
                                label="Chunk strategy",
                                render=False,
                            )
                            with gr.Row():
                                example_imgs_backer = gr.Button("Go back", visible=False)
                                example_imgs_confirmer = gr.Button("Confirm", visible=False)
                            example_imgs.select(
                                get_examples,
                                outputs=[
                                    example_imgs_expander,
                                    example_imgs_confirmer,
                                    example_imgs_backer,
                                    example_imgs,
                                ],
                            )
                            example_imgs_confirmer.click(
                                lambda x: (
                                    x,
                                    gr.update(visible=False),
                                    gr.update(visible=False),
                                    gr.update(visible=False),
                                    gr.update(visible=True),
                                ),
                                inputs=[example_imgs_expander],
                                outputs=[
                                    input_imgs,
                                    example_imgs_expander,
                                    example_imgs_confirmer,
                                    example_imgs_backer,
                                    example_imgs,
                                ],
                            )
                            example_imgs_backer.click(
                                lambda: (
                                    gr.update(visible=False),
                                    gr.update(visible=False),
                                    gr.update(visible=False),
                                    gr.update(visible=True),
                                ),
                                outputs=[
                                    example_imgs_expander,
                                    example_imgs_confirmer,
                                    example_imgs_backer,
                                    example_imgs,
                                ],
                            )
                            preprocessed = gr.State()
                            shorter = gr.Number(
                                value=576, label="Resize", render=False, scale=3
                            )
                            keep_aspect = gr.Checkbox(
                                True, label="Keep aspect ratio", render=False, scale=1
                            )
                            preprocess_btn.click(
                                lambda r, *args: r.preprocess(*args),
                                inputs=[renderer, input_imgs, shorter, keep_aspect],
                                outputs=[preprocessed, preprocess_progress, chunk_strategy],
                            )
                            preprocess_btn.click(
                                lambda: gr.update(visible=True), outputs=[preprocess_progress]
                            )
                            preprocessed.change(
                                lambda r, *args: r.visualize_scene(*args),
                                inputs=[renderer, preprocessed],
                            )
                        with gr.Row():
                            shorter.render()
                            keep_aspect.render()
                        with gr.Row():
                            seed = gr.Number(value=23, label="Random seed")
                            chunk_strategy.render()
                            cfg = gr.Slider(2.0, 10.0, value=3.0, label="CFG value")
                        camera_scale = gr.Slider(
                            0.1,
                            10.0,
                            value=2.0,
                            label="Camera scale (useful for single image case)",
                        )
                    with gr.Column():
                        with gr.Group():
                            render_btn.render()
                            render_progress = gr.Textbox(
                                label="", visible=False, interactive=False
                            )
                        output_video = gr.Video(
                            label="Output", interactive=False, autoplay=True, loop=True
                        )
                        render_btn.click(
                            lambda r, *args: (yield from r.render(*args)),
                            inputs=[
                                renderer,
                                preprocessed,
                                seed,
                                chunk_strategy,
                                cfg,
                                camera_scale,
                            ],
                            outputs=[output_video, render_progress],
                        )
                        render_btn.click(
                            lambda: gr.update(visible=True), outputs=[render_progress]
                        )
        # Attach a callback using the tab select API (as described in https://www.gradio.app/docs/gradio/tab#tab-select)
        # to load the Advanced tab server only once when the tab is selected.
        advanced_loaded = gr.State(value=False)
        def maybe_load_server(req: gr.Request, loaded, evt: gr.SelectData):
            if evt.value == "Advanced" and not loaded:
                # Call start_server with the current request since it's triggered by tab selection.
                renderer_obj, viewport_obj = start_server(req)
                return renderer_obj, viewport_obj, True
            else:
                return gr.update(), gr.update(), loaded
        tabs.select(
            maybe_load_server,
            inputs=[advanced_loaded],
            outputs=[renderer, viewport, advanced_loaded]
        )
        demo.unload(stop_server)
    demo.launch(
        share=share,
        server_port=server_port,
        show_error=True,
        allowed_paths=[WORK_DIR, EXAMPLE_DIR],
    )

if __name__ == "__main__":
    tyro.cli(main)
