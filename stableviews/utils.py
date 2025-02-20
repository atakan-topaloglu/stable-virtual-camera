import os
import safetensors.torch
import torch

from stableviews.model import StableViews, StableViewsParams
from huggingface_hub import hf_hub_download


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_model(
    pretrained_model_name_or_path: str, weight_name: str,device: str | torch.device = "cuda", verbose: bool = False
) -> StableViews:

    if os.path.isdir(pretrained_model_name_or_path):
        weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
    else:
        weight_path = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=weight_name
        )
        config_path = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename="config.yaml"
        )

    state_dict = safetensors.torch.load_file(
        weight_path,
        device=str(device),
    )
    model_state_dict = {
        k.removeprefix("model.diffusion_model."): v
        for k, v in state_dict.items()
        if k.startswith("model.diffusion_model.")
    }

    with torch.device("meta"):
        model = StableViews(StableViewsParams()).to(torch.bfloat16)

    missing, unexpected = model.load_state_dict(
        model_state_dict, strict=False, assign=True
    )
    if verbose:
        print_load_warning(missing, unexpected)
    return model
