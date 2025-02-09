import safetensors.torch
import torch

from scena.model import Scena, ScenaParams


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_model(device: str | torch.device = "cuda", verbose: bool = False) -> Scena:
    state_dict = safetensors.torch.load_file(
        "/admin/home-hangg/projects/stable-research/logs/inference/3d_diffusion-jensen-3d_attn-all1_img2vid25_FT21drunk_plucker_concat_norm_mv_cat3d_v3_discrete_no-clip-txt_3d-attn-with-view-attn-mixing_freeze-pretrained_5355784/epoch=000000-step=000600000_inference.safetensors",
        device=str(device),
    )
    model_state_dict = {
        k.removeprefix("model.diffusion_model."): v
        for k, v in state_dict.items()
        if k.startswith("model.diffusion_model.")
    }

    with torch.device("meta"):
        model = Scena(ScenaParams()).to(torch.bfloat16)

    missing, unexpected = model.load_state_dict(
        model_state_dict, strict=False, assign=True
    )
    if verbose:
        print_load_warning(missing, unexpected)
    return model
