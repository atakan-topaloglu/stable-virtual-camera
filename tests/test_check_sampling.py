import os
import os.path as osp
import sys

import imageio.v3 as iio
import numpy as np
import torch

from stableviews.model import SGMWrapper
from stableviews.modules.autoencoder import AutoEncoder
from stableviews.sampling import (
    DDPMDiscretization,
    DiscreteDenoiser,
    EulerEDMSampler,
    MultiviewCFG,
)
from stableviews.utils import load_model

sys.path.insert(0, "/admin/home-hangg/projects/stable-research/")

from scripts.threeD_diffusion.run_eval import seed_everything

device = torch.device("cuda:0")
work_dir = "work_dirs/_tests/test_check_sampling/"
os.makedirs(work_dir, exist_ok=True)

value_dict = torch.load("tests/sampling_value_dict.pth", map_location=device)

steps = 50
s_churn = 0.0
s_tmin = 0.0
s_tmax = 999.0
s_noise = 1.0
# discretization_config = {
#     "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
#     "params": {
#         "linear_start": 5e-06,
#         "linear_end": 0.012,
#         "schedule": "linear",
#         "log_snr_shift": 2.4,
#     },
# }
# guider_config = [
#     {
#         "target": "sgm.modules.diffusionmodules.guiders.MultiPredictionGuider",
#         "params": {
#             "max_scale": 3.0,
#             "min_scale": 1.2,
#             "additional_cond_keys": ["replace", "dense_vector"],
#             "scale_rule_config": {
#                 "target": "sgm.modules.diffusionmodules.guiders.LowCFGCloseInputFrame",
#                 "params": {"scale_low": 1.2},
#             },
#             "scale_schedule_config": {
#                 "target": "sgm.modules.diffusionmodules.guiders.IdentitySchedule",
#                 "params": {"scale": 3.0},
#             },
#             "dyn_thresh_config": {
#                 "target": "sgm.modules.diffusionmodules.guiders.NoDynamicThresholding"
#             },
#         },
#     },
#     {
#         "target": "sgm.modules.diffusionmodules.guiders.MultiTrianglePredictionGuider",
#         "params": {
#             "max_scale": 2.0,
#             "num_frames": 21,
#             "min_scale": 1.2,
#             "orbit": False,
#             "self_norm": True,
#             "additional_cond_keys": ["replace", "dense_vector"],
#             "scale_rule_config": {
#                 "target": "sgm.modules.diffusionmodules.guiders.LowCFGCloseInputFrame",
#                 "params": {"scale_low": 1.2},
#             },
#             "scale_schedule_config": {
#                 "target": "sgm.modules.diffusionmodules.guiders.IdentitySchedule",
#                 "params": {"scale": 2.0},
#             },
#             "dyn_thresh_config": {
#                 "target": "sgm.modules.diffusionmodules.guiders.NoDynamicThresholding"
#             },
#         },
#     },
# ][0]
# sampler_sgm = EulerEDMSamplerSGM(
#     num_steps=steps,
#     discretization_config=discretization_config,
#     guider_config=guider_config,
#     enforce_clean_at_cond_frame=False,
#     s_churn=s_churn,
#     s_tmin=s_tmin,
#     s_tmax=s_tmax,
#     s_noise=s_noise,
#     verbose=True,
#     device=device,
# )
# denoiser_sgm = DiscreteDenoiserSGM(
#     discretization_config={
#         "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
#     },
#     num_idx=1000,
#     scaling_config={
#         "target": "sgm.modules.diffusionmodules.denoiser_scaling.EpsScaling"
#     },
# ).to(device)
# version_dict, engine = init_model(
#     version="prediction_3D_SD21V_discrete_plucker_norm_replace",
#     config="/admin/home-hangg/projects/stable-research/configs/3d_diffusion/jensen/inference/sd_3d-view-attn_21FT_discrete_no-clip-txt_pl---nk_plucker_concat_norm_mv_cat3d_v3_discrete_no-clip-txt_3d-attn-with-view-attn-mixing_freeze-pretrained_5355784_ckpt600000.yaml",
# )
# model_sgm = engine.model
# # seed_everything(0)
# # with torch.inference_mode(), torch.autocast(
# #     device_type=device.type, dtype=torch.float16
# # ):
# #     denoiser_sgm.discretization = sampler_sgm.discretization
# #     denoiser_sgm.register_sigmas()
# #     samples_z = sampler_sgm(
# #         lambda input, sigma, c: denoiser_sgm(
# #             model_sgm, input, sigma, c, **value_dict["additional_model_inputs"]
# #         ),
# #         value_dict["randn"],
# #         cond=value_dict["c"],
# #         uc=value_dict["uc"],
# #         verbose=True,
# #         **value_dict["additional_sampler_inputs"],
# #     )
# # samples = engine.decode_first_stage(samples_z)
# # iio.imwrite(
# #     osp.join(work_dir, "sgm.mp4"),
# #     (
# #         (samples.permute(0, 2, 3, 1) * 0.5 + 0.5).clamp(0, 1).cpu().numpy() * 255.0
# #     ).astype(np.uint8),
# #     fps=5,
# # )
# # # __import__("ipdb").set_trace()
#
model = load_model("cpu", verbose=True).eval()
model_wrapped = SGMWrapper(model).to(device)
# seed_everything(0)
# with torch.inference_mode(), torch.autocast(
#     device_type=device.type, dtype=torch.float16
# ):
#     denoiser_sgm.discretization = sampler_sgm.discretization
#     denoiser_sgm.register_sigmas()
#     samples_z = sampler_sgm(
#         lambda input, sigma, c: denoiser_sgm(
#             model_wrapped,
#             input,
#             sigma,
#             c,
#             num_frames=value_dict["additional_model_inputs"]["num_video_frames"],
#         ),
#         value_dict["randn"],
#         cond=value_dict["c"],
#         uc=value_dict["uc"],
#         verbose=True,
#         **value_dict["additional_sampler_inputs"],
#     )
# samples = engine.decode_first_stage(samples_z)
# iio.imwrite(
#     osp.join(work_dir, "sgm_wrapper.mp4"),
#     (
#         (samples.permute(0, 2, 3, 1) * 0.5 + 0.5).clamp(0, 1).cpu().numpy() * 255.0
#     ).astype(np.uint8),
#     fps=5,
# )

ae = AutoEncoder(chunk_size=1).to(device)
discretization = DDPMDiscretization()
guider = MultiviewCFG()
denoiser = DiscreteDenoiser(discretization=discretization, num_idx=1000, device=device)
sampler = EulerEDMSampler(
    discretization=discretization,
    guider=guider,
    num_steps=steps,
    verbose=True,
    device=device,
    s_churn=s_churn,
    s_tmin=s_tmin,
    s_tmax=s_tmax,
    s_noise=s_noise,
)
seed_everything(0)
with (
    torch.inference_mode(),
    torch.autocast(device_type=device.type, dtype=torch.float16),
):
    samples_z = sampler(
        lambda input, sigma, c: denoiser(
            model_wrapped,
            input,
            sigma,
            c,
            num_frames=value_dict["additional_model_inputs"]["num_video_frames"],
        ),
        value_dict["randn"],
        scale=3.0,
        cond=value_dict["c"],
        uc=value_dict["uc"],
        verbose=True,
        **{
            k: value_dict["additional_sampler_inputs"][k]
            for k in ["c2w", "K", "input_frame_mask"]
        },
    )
samples = ae.decode(samples_z)
iio.imwrite(
    osp.join(work_dir, "stableviews.mp4"),
    (
        (samples.permute(0, 2, 3, 1) * 0.5 + 0.5).clamp(0, 1).cpu().numpy() * 255.0
    ).astype(np.uint8),
    fps=5,
)
__import__("ipdb").set_trace()
