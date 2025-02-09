import copy
import os
import os.path as osp
import sys
from dataclasses import dataclass
from typing import Literal

import imageio.v3 as iio
import numpy as np
import torch
from einops import rearrange, repeat
from tqdm import tqdm

sys.path.insert(0, "/admin/home-hangg/projects/stable-research/")
from scripts.threeD_diffusion.run_eval_gr_hoster import (
    CovariancePredictionGuider,
    LinearPredictionGuider,
    LinearTrianglePredictionGuider,
    MultiPredictionGuider,
    MultiTrianglePredictionGuider,
    PostHocDecoderWithTime,
    TrapezoidPredictionGuider,
    VanillaCFG,
    VideoPredictionEmbedderWithEncoder,
    chunk_input_and_test,
    decode_output,
    default,
    extend_dict,
    get_arc_horizontal_w2cs,
    get_default_intrinsics,
    get_k_from_dict,
    get_plucker_coordinates,
    get_unique_embedder_keys_from_conditioner,
    init_embedder_options_no_st,
    init_model,
    init_sampling_no_st,
    load_model,
    seed_everything,
    to_hom_pose,
    transform_img_and_K,
    unload_model,
    update_kv_for_dict,
)


@dataclass
class ScenaData:
    input_imgs: torch.Tensor
    input_c2ws: torch.Tensor
    input_Ks: torch.Tensor
    target_c2ws: torch.Tensor
    target_Ks: torch.Tensor
    anchor_c2ws: torch.Tensor
    anchor_Ks: torch.Tensor


# Constants.
context_window = 21
target_wh = (576, 576)
device = "cuda:0"

# Args.
img_path = "/admin/home-hangg/projects/scena-release/assets/images/lily-dragon.png"
output_dir = "work_dirs/_tests/test_sgm_single_img/lily-dragon/360v1/"
num_target_frames = 80
chunk_strategy: Literal["nearest", "interp", "interp-gt"] = "interp-gt"
chunk_strategy_first_pass = "gt_nearest"
seed = 23
options = {
    "as_shuffled": True,
    "discretization": 0,
    "beta_linear_start": 5e-06,
    "beta_linear_end": 0.012,
    "beta_schedule": "linear",
    "log_snr_shift": 2.4,
    "guider": (6, 5),
    "cfg": (3.0, 2.0),
    "cfg_min": 1.2,
    "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
    "num_steps": 50,
    "camera_mode": "plucker",
    "input_frame_per_output": True,
    "additional_guider_kwargs": {
        "additional_cond_keys": ["replace", "dense_vector"],
        "scale_rule_config": {
            "target": "sgm.modules.diffusionmodules.guiders.LowCFGCloseInputFrame",
            "params": {"scale_low": 1.2},
        },
        "scale_schedule_config": {
            "target": "sgm.modules.diffusionmodules.guiders.IdentitySchedule",
            "params": {"scale": 2.0},
        },
        "dyn_thresh_config": {
            "target": "sgm.modules.diffusionmodules.guiders.NoDynamicThresholding"
        },
    },
    "dataset_latent": False,
    "use_original_benchmark_img_wh": False,
    "wide_aspect_ratio_mode": "gen_large_then_crop",
    "camera_scale": 2.0,
    "camera_center": "mass-global",
    "camera_intrinsics": True,
    "last_frame_as_source": False,
    "num_input_frames": 1,
    "num_cameras": -1,
    "traj_prior": "360v1",
    "num_ndv_frames": num_target_frames,
    "video_save_fps": 30,
    "chunk_strategy": "interp-gt",
    "num_frames": context_window,
}

# Get scena data.
img = iio.imread(img_path)
num_input_frames = 1
input_imgs = repeat(
    torch.as_tensor(img) / 255.0, "h w c -> n c h w", n=num_input_frames
)
input_imgs = transform_img_and_K(input_imgs, None, *target_wh)[0]
input_c2ws = repeat(torch.eye(4), "i j -> n i j", n=num_input_frames)
input_Ks = repeat(get_default_intrinsics(), "1 i j -> n i j", n=num_input_frames)
target_c2ws = torch.linalg.inv(
    get_arc_horizontal_w2cs(
        torch.linalg.inv(input_c2ws[0]),
        torch.tensor([0, 0, 10]),
        None,
        num_target_frames,
    )
)
target_Ks = repeat(get_default_intrinsics(), "1 i j -> n i j", n=num_target_frames)
if chunk_strategy.startswith("interp"):
    num_anchor_frames = max(
        np.ceil(
            num_target_frames
            / (
                context_window
                - 2
                - (num_input_frames if chunk_strategy == "interp-gt" else 0)
            )
        )
        .astype(int)
        .item()
        + 1,
        context_window - num_input_frames,
    )
    include_start_end = True
else:
    num_anchor_frames = max(context_window - num_input_frames, 0)
    include_start_end = False
if include_start_end:
    anchor_c2ws = torch.linalg.inv(
        get_arc_horizontal_w2cs(
            torch.eye(4),
            torch.tensor([0, 0, 10]),
            None,
            num_anchor_frames,
            endpoint=True,
        )
    )
else:
    anchor_c2ws = torch.linalg.inv(
        get_arc_horizontal_w2cs(
            torch.eye(4),
            torch.tensor([0, 0, 10]),
            None,
            num_anchor_frames + 1,
            endpoint=False,
        )
    )[1:]
anchor_Ks = repeat(get_default_intrinsics(), "1 i j -> n i j", n=num_anchor_frames)

input_indices = list(range(num_input_frames))
anchor_indices = np.linspace(
    1,
    num_target_frames,
    num_anchor_frames + 1 - include_start_end,
    endpoint=include_start_end,
)[1 - include_start_end :].tolist()
target_indices = np.arange(
    num_input_frames, num_input_frames + num_target_frames
).tolist()

data = ScenaData(
    input_imgs.to(device),
    input_c2ws.to(device),
    input_Ks.to(device),
    target_c2ws.to(device),
    target_Ks.to(device),
    anchor_c2ws.to(device),
    anchor_Ks.to(device),
)

# Load model.
version_dict, engine = init_model(
    version="prediction_3D_SD21V_discrete_plucker_norm_replace",
    config="/admin/home-hangg/projects/stable-research/configs/3d_diffusion/jensen/inference/sd_3d-view-attn_21FT_discrete_no-clip-txt_pl---nk_plucker_concat_norm_mv_cat3d_v3_discrete_no-clip-txt_3d-attn-with-view-attn-mixing_freeze-pretrained_5355784_ckpt600000.yaml",
)
conditioner = engine.conditioner


# Run inference.
embedder_keys = set(get_unique_embedder_keys_from_conditioner(conditioner))


def get_value_dict(
    curr_imgs,
    curr_imgs_clip,
    curr_input_frame_indices,
    curr_c2ws,
    curr_Ks,
    curr_input_camera_indices,
    curr_depths,
    all_c2ws,
    as_shuffled=True,
):
    assert sorted(curr_input_camera_indices) == sorted(
        range(len(curr_input_camera_indices))
    )
    H, W, T, F = curr_imgs.shape[-2], curr_imgs.shape[-1], len(curr_imgs), 8

    value_dict = init_embedder_options_no_st(
        embedder_keys,
        prompt="A scene.",
        negative_prompt="",
    )
    value_dict["image_only_indicator"] = int(as_shuffled)

    value_dict["cond_frames_without_noise"] = curr_imgs_clip[curr_input_frame_indices]
    T = len(curr_imgs)
    value_dict["cond_frames_mask"] = torch.zeros(T, dtype=torch.bool)
    value_dict["cond_frames_mask"][curr_input_frame_indices] = True
    value_dict["cond_frames"] = curr_imgs
    value_dict["cond_aug"] = 0.0

    camera_scale = 2.0
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
        if torch.isclose(torch.norm(camera_dists[0]), torch.zeros(1)).any()
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
    input_indices: list[int],
    test_indices: list[int],
    padding_mode: Literal["first", "last", "none"] = "last",
):
    T = context_window
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
        sorted_inds = np.argsort(input_indices)
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
    input,
    test,
    input_maps,
    test_maps,
):
    # Support padding for legacy reason, the right way is to do the attention masking.
    T = len(input_maps)
    assembled = torch.zeros_like(test[-1:]).repeat_interleave(T, dim=0)
    assembled[input_maps != -1] = input[input_maps[input_maps != -1]]
    assembled[test_maps != -1] = test[test_maps[test_maps != -1]]
    assert np.logical_xor(input_maps != -1, test_maps != -1).all()
    return assembled


def get_batch(
    keys,
    value_dict: dict,
    N: list[int],
    device: str = "cuda",
    T: int = context_window,
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


def do_sample(
    model,
    sampler_,
    value_dict,
    H,
    W,
    C,
    F,
    num_samples: int = 1,
    force_uc_zero_embeddings: list[str] = [],
    force_cond_zero_embeddings: list[str] = [],
    batch2model_input: list[str] = [],
    T=None,
    additional_batch_uc_fields: list[str] = [],
    encoding_t=None,
    decoding_t=None,
    log_groundtruth=False,
    keys_to_log_from_gt=(),
    log_gt_frames=False,
    log_processed=False,
    dataset_latent=False,
    **_,
):
    if encoding_t:
        for emb in model.conditioner.embedders:
            if isinstance(emb, VideoPredictionEmbedderWithEncoder):
                emb.en_and_decode_n_samples_a_time = encoding_t
    if decoding_t:
        model.en_and_decode_n_samples_a_time = decoding_t

    precision_scope = torch.autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                if T is not None:
                    num_samples = [num_samples, T]
                else:
                    num_samples = [num_samples]

                load_model(model.conditioner)
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    num_samples,
                    T=T,
                    additional_batch_uc_fields=additional_batch_uc_fields,
                )

                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                    force_cond_zero_embeddings=force_cond_zero_embeddings,
                    n_cond_frames=T,
                )
                unload_model(model.conditioner)

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: np.prod(num_samples).item()].to("cuda"),
                            (c, uc),
                        )

                additional_model_inputs = {}
                for k in batch2model_input:
                    if k == "image_only_indicator":
                        assert T is not None

                        if isinstance(
                            sampler_.guider,
                            (
                                VanillaCFG,
                                LinearPredictionGuider,
                                LinearTrianglePredictionGuider,
                                TrapezoidPredictionGuider,
                                MultiTrianglePredictionGuider,
                                MultiPredictionGuider,
                                CovariancePredictionGuider,
                            ),
                        ):
                            additional_model_inputs[k] = torch.full(
                                (num_samples[0] * 2, num_samples[1]),
                                value_dict[k],
                            ).to("cuda")
                        else:
                            additional_model_inputs[k] = (
                                value_dict[k].repeat(num_samples).to("cuda")
                            )
                    else:
                        additional_model_inputs[k] = batch[k]

                shape = (np.prod(num_samples).item(), C, H // F, W // F)
                randn = torch.randn(shape).to("cuda")

                #####
                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                # THIS IS VERY IMPORTANT! IF your denoiser is discrete denoiser
                # that requries discretization, you should make sure the one used
                # is the same one used for sampler. This is to avoid potential
                # inconsistency between the training and inference config.
                if hasattr(model.denoiser, "discretization"):
                    model.denoiser.discretization = sampler_.discretization
                    model.denoiser.register_sigmas()
                #######

                additional_sampler_inputs = {
                    "c2w": value_dict["c2w"].to("cuda"),
                    "K": value_dict["K"].to("cuda"),
                    "input_frame_mask": value_dict["cond_frames_mask"].to("cuda"),
                }

                load_model(model.denoiser)
                load_model(model.model)
                samples_z = sampler_(
                    denoiser, randn, cond=c, uc=uc, **additional_sampler_inputs
                )
                unload_model(model.model)
                unload_model(model.denoiser)

                load_model(model.first_stage_model)
                if isinstance(model.first_stage_model.decoder, PostHocDecoderWithTime):
                    samples = model.decode_first_stage(
                        samples_z, timesteps=default(decoding_t, T)
                    )
                else:
                    samples = model.decode_first_stage(samples_z)

                if isinstance(samples, dict) and model.postprocessor_for_logging:
                    if log_groundtruth:
                        samples_new = model.postprocessor_for_logging(
                            batch,
                            {"samples": samples},
                            n=np.prod(num_samples).item(),
                            log_from_batch=True,
                            log_raw=True,
                            log_processed=log_processed,
                        )
                        for k in (
                            keys_to_log_from_gt
                            if isinstance(keys_to_log_from_gt, (tuple, list))
                            else keys_to_log_from_gt.split(",")
                        ):
                            if k in batch:
                                samples_new[k] = batch[k]
                    else:
                        samples.update(
                            model.postprocessor_for_logging(
                                {"num_video_frames": T, **samples},
                                n=np.prod(num_samples).item(),
                                log_from_batch=True,
                                log_raw=True,
                                log_processed=log_processed,
                            )
                        )
                        samples_new = {}

                    for k, v in samples.items():
                        samples_new[f"samples-{k}"] = v
                    samples = samples_new

                if log_gt_frames:
                    if dataset_latent:
                        # if batched data is with latent, then decode it
                        value_dict["cond_frames"] = model.decode_first_stage(
                            model.encode_first_stage(
                                value_dict["cond_frames"].to("cuda"),
                                sampled_modality_ids=[0],
                            ),
                            sampled_modality_ids=[0],
                        )["rgb"]  # (T, 3, H, W)

                    if isinstance(samples, dict):
                        samples_new = {}
                        for k, v in samples.items():
                            samples_new[f"{k}/image" if "rgb" in k else k] = samples[k]
                        samples = samples_new
                    else:
                        samples = {
                            "samples-rgb/image": samples,
                        }
                    samples.update({"rgb/image": value_dict["cond_frames"]})

                unload_model(model.first_stage_model)
                return samples


input_samples = rearrange(
    input_imgs.cpu().numpy() * 255.0,
    "n c h w -> n h w c",
).astype(np.uint8)
input_dir = osp.join(output_dir, "input")
os.makedirs(input_dir, exist_ok=True)
for i, img in enumerate(input_samples):
    iio.imwrite(osp.join(input_dir, f"{i:04d}.png"), img)
input_path = osp.join(output_dir, "input.mp4")
iio.imwrite(input_path, input_samples, fps=2.0)


samplers = init_sampling_no_st(options=options)

seed_everything(seed)

(
    _,
    input_inds_per_chunk,
    input_sels_per_chunk,
    anchor_inds_per_chunk,
    anchor_sels_per_chunk,
) = chunk_input_and_test(
    context_window,
    input_c2ws,
    anchor_c2ws,
    input_indices,
    anchor_indices,
    task="img2trajvid",
    chunk_strategy=chunk_strategy_first_pass,
    gt_input_inds=list(range(num_input_frames)),
    verbose=True,
)
print(
    f"Two passes (first) - chunking with `{chunk_strategy_first_pass}` strategy: total "
    f"{len(input_inds_per_chunk)} forward(s) ..."
)

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
    ) = pad_indices(
        chunk_input_sels,
        chunk_anchor_sels,
        padding_mode="last",
    )
    curr_imgs, curr_imgs_clip, curr_c2ws, curr_Ks, curr_depths = [
        assemble(
            input=x[chunk_input_inds],
            test=y[chunk_anchor_inds],
            input_maps=curr_input_maps,
            test_maps=curr_anchor_maps,
        )
        for x, y in zip(
            [
                torch.cat(
                    [
                        input_imgs * 2.0 - 1.0,
                        get_k_from_dict(all_samples, "samples-rgb").to(
                            input_imgs.device
                        ),
                    ],
                    dim=0,
                ),
                torch.cat(
                    [
                        input_imgs * 2.0 - 1.0,
                        get_k_from_dict(all_samples, "samples-rgb").to(
                            input_imgs.device
                        ),
                    ],
                    dim=0,
                ),
                torch.cat([input_c2ws, anchor_c2ws[all_anchor_inds]], dim=0),
                torch.cat([input_Ks, anchor_Ks[all_anchor_inds]], dim=0),
                torch.cat([input_Ks, anchor_Ks[all_anchor_inds]], dim=0),
            ],  # procedually append generated prior views to the input views
            [
                repeat(
                    torch.zeros_like(input_imgs[:1]),
                    "1 c h w -> n c h w",
                    n=num_anchor_frames,
                ),
                repeat(
                    torch.zeros_like(input_imgs[:1]),
                    "1 c h w -> n c h w",
                    n=num_anchor_frames,
                ),
                anchor_c2ws,
                anchor_Ks,
                anchor_Ks,
            ],
        )
    ]
    value_dict = get_value_dict(
        curr_imgs,
        curr_imgs_clip,
        curr_input_sels,
        curr_c2ws,
        curr_Ks,
        list(range(context_window)),
        curr_depths,
        all_c2ws=torch.cat([input_c2ws, target_c2ws], 0),
        as_shuffled=(
            options["as_shuffled"]
            and any(
                map(
                    lambda x: x in chunk_strategy_first_pass,
                    ["nearest", "gt"],
                )
            )  # "interp" not in chunk_strategy_first_pass
        ),
    )
    samples = do_sample(
        engine,
        (
            samplers[1]
            if len(samplers) > 1
            and options.get("ltr_first_pass", False)
            and chunk_strategy_first_pass != "gt"
            and i > 0
            else samplers[0]
        ),
        value_dict,
        target_wh[1],
        target_wh[0],
        4,
        8,
        T=context_window,
        batch2model_input=["num_video_frames", "image_only_indicator"],
        **options,
    )
    samples = decode_output(samples, context_window, chunk_anchor_sels)
    extend_dict(all_samples, samples)
    all_anchor_inds.extend(chunk_anchor_inds)

first_pass_samples = rearrange(
    (all_samples["samples-rgb/image"] / 2.0 + 0.5).clamp(0.0, 1.0).cpu().numpy()
    * 255.0,
    "n c h w -> n h w c",
).astype(np.uint8)
first_pass_dir = osp.join(output_dir, "first_pass")
os.makedirs(first_pass_dir, exist_ok=True)
for i, img in enumerate(first_pass_samples):
    iio.imwrite(osp.join(first_pass_dir, f"{i:04d}.png"), img)
first_pass_path = osp.join(output_dir, "first_pass.mp4")
iio.imwrite(first_pass_path, first_pass_samples, fps=5.0)

assert (
    anchor_indices is not None
), "`anchor_frame_indices` needs to be set if using 2-pass sampling."
anchor_argsort = np.argsort(input_indices + anchor_indices).tolist()
anchor_indices = np.array(input_indices + anchor_indices)[anchor_argsort].tolist()
gt_input_inds = [anchor_argsort.index(i) for i in range(input_c2ws.shape[0])]

anchor_imgs = torch.cat(
    [input_imgs, get_k_from_dict(all_samples, "samples-rgb") / 2.0 + 0.5], dim=0
)[anchor_argsort]
anchor_c2ws = torch.cat([input_c2ws, anchor_c2ws], dim=0)[anchor_argsort]
anchor_Ks = torch.cat([input_Ks, anchor_Ks], dim=0)[anchor_argsort]

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
    context_window,
    anchor_c2ws,
    target_c2ws,
    anchor_indices,
    target_indices,
    task="img2trajvid",
    chunk_strategy=chunk_strategy,
    gt_input_inds=gt_input_inds,
    verbose=options.get("sampler_verbose", True),
)
print(
    f"Two passes (second) - chunking with `{chunk_strategy}` strategy: total "
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
    ) = pad_indices(
        chunk_anchor_sels,
        chunk_target_sels,
        padding_mode="last",
    )
    curr_imgs, curr_imgs_clip, curr_c2ws, curr_Ks, curr_depths = [
        assemble(
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
                    torch.zeros_like(input_imgs[:1]),
                    "1 c h w -> n c h w",
                    n=num_target_frames,
                ),
                repeat(
                    torch.zeros_like(input_imgs[:1]),
                    "1 c h w -> n c h w",
                    n=num_target_frames,
                ),
                target_c2ws,
                target_Ks,
                target_Ks,
            ],
        )
    ]
    value_dict = get_value_dict(
        curr_imgs,
        curr_imgs_clip,
        curr_anchor_sels,
        curr_c2ws,
        curr_Ks,
        list(range(context_window)),
        curr_depths,
        all_c2ws=torch.cat([input_c2ws, target_c2ws], 0),
        as_shuffled=(
            options["as_shuffled"]
            and any(
                map(
                    lambda x: x in chunk_strategy,
                    ["nearest", "gt"],
                )
            )  # "interp" not in chunk_strategy
        ),
    )
    samples = do_sample(
        engine,
        samplers[1] if len(samplers) > 1 else samplers[0],
        value_dict,
        target_wh[1],
        target_wh[0],
        4,
        8,
        T=context_window,
        batch2model_input=["num_video_frames", "image_only_indicator"],
        **options,
    )
    samples = decode_output(samples, context_window, chunk_target_sels)
    extend_dict(all_samples, samples)
    all_target_inds.extend(chunk_target_inds)

all_samples = {
    key: value[np.argsort(all_target_inds)] for key, value in all_samples.items()
}
second_pass_samples = rearrange(
    (all_samples["samples-rgb/image"] / 2.0 + 0.5).clamp(0.0, 1.0).cpu().numpy()
    * 255.0,
    "n c h w -> n h w c",
).astype(np.uint8)
second_pass_dir = osp.join(output_dir, "second_pass")
os.makedirs(second_pass_dir, exist_ok=True)
for i, img in enumerate(second_pass_samples):
    iio.imwrite(osp.join(second_pass_dir, f"{i:04d}.png"), img)
second_pass_path = osp.join(output_dir, "second_pass.mp4")
iio.imwrite(second_pass_path, second_pass_samples, fps=30.0)
