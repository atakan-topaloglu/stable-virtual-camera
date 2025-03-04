import sys

import torch
import torch.nn.functional as F
from einops import repeat

from stableviews.modules.autoencoder import AutoEncoder
from stableviews.modules.conditioner import CLIPConditioner

sys.path.insert(0, "/admin/home-hangg/projects/stable-research/")
from scripts.threeD_diffusion.run_eval import (
    get_batch,
    get_unique_embedder_keys_from_conditioner,
    init_model,
)

device = torch.device("cuda:0")
T = 21

value_dict = torch.load("tests/conditioner_value_dict.pth", map_location=device)

version_dict, engine = init_model(
    version="prediction_3D_SD21V_discrete_plucker_norm_replace",
    config="/admin/home-hangg/projects/stable-research/configs/3d_diffusion/jensen/inference/sd_3d-view-attn_21FT_discrete_no-clip-txt_pl---nk_plucker_concat_norm_mv_cat3d_v3_discrete_no-clip-txt_3d-attn-with-view-attn-mixing_freeze-pretrained_5355784_ckpt600000.yaml",
)
conditioner_sgm = engine.conditioner
# with torch.inference_mode(), torch.autocast(
#     device_type=device.type,
#     dtype=torch.float16,  # Note that this has to be f16 to match single image script.
# ):
with (
    torch.inference_mode(),
    torch.autocast(device_type=device.type, dtype=torch.bfloat16),
):
    batch, batch_uc = get_batch(
        get_unique_embedder_keys_from_conditioner(conditioner_sgm),
        value_dict,
        [1, T],
        T=T,
        additional_batch_uc_fields=[],
    )
    c, uc = conditioner_sgm.get_unconditional_conditioning(
        batch,
        batch_uc=batch_uc,
        force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"],
        force_cond_zero_embeddings=[],
        n_cond_frames=T,
    )

# (T, 3, H, W).
imgs = value_dict["cond_frames"]
# (T,), bool.
input_masks = value_dict["cond_frames_mask"]
# (T, 6, H // 8, W // 8).
pluckers = value_dict["plucker_coordinate"]

clip_conditioner = CLIPConditioner().to(device)
with (
    torch.inference_mode(),
    torch.autocast(device_type=device.type, dtype=torch.bfloat16),
):
    c_crossattn = clip_conditioner(imgs[input_masks]).mean(0)
    uc_crossattn = torch.zeros_like(c_crossattn)
assert torch.allclose(c["crossattn"], c_crossattn.float()), __import__(
    "ipdb"
).set_trace()
assert torch.allclose(uc["crossattn"], uc_crossattn.float()), __import__(
    "ipdb"
).set_trace()

ae = AutoEncoder().to(device)
with (
    torch.inference_mode(),
    torch.autocast(device_type=device.type, dtype=torch.bfloat16),
):
    input_latents = F.pad(ae.encode(imgs[input_masks]), (0, 0, 0, 0, 0, 1), value=1.0)
    c_replace = input_latents.new_zeros(T, *input_latents.shape[1:])
    c_replace[input_masks] = input_latents
    uc_replace = torch.zeros_like(c_replace)
# TODO(hangg): This one doesn't work, probably because of the precision.
# assert torch.allclose(c["replace"], c_replace.float()), __import__("ipdb").set_trace()
print(torch.allclose(c["replace"], c_replace.float()))
assert torch.allclose(uc["replace"], uc_replace.float()), __import__("ipdb").set_trace()

c_concat = torch.cat(
    [
        repeat(input_masks, "n -> n 1 h w", h=pluckers.shape[2], w=pluckers.shape[3]),
        pluckers,
    ],
    1,
)
uc_concat = torch.cat([pluckers.new_zeros(T, 1, *pluckers.shape[-2:]), pluckers], 1)
assert torch.allclose(c["concat"], c_concat.float()), __import__("ipdb").set_trace()
assert torch.allclose(uc["concat"], uc_concat.float()), __import__("ipdb").set_trace()

c_dense_vector = pluckers
uc_dense_vector = c_dense_vector
assert torch.allclose(c["dense_vector"], c_dense_vector.float()), __import__(
    "ipdb"
).set_trace()
assert torch.allclose(uc["dense_vector"], uc_dense_vector.float()), __import__(
    "ipdb"
).set_trace()
__import__("ipdb").set_trace()
