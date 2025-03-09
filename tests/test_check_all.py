import copy
import os
import os.path as osp

import imageio.v3 as iio
import numpy as np
import torch

from seva.eval import do_sample, get_value_dict
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import (
    DDPMDiscretization,
    DiscreteDenoiser,
    EulerEDMSampler,
    MultiviewCFG,
)
from seva.utils import load_model

device = torch.device("cuda:0")
work_dir = "work_dirs/_tests/test_check_all/"
os.makedirs(work_dir, exist_ok=True)

data = torch.load("tests/all_value_dict.pth", map_location="cpu")

steps = 50
s_churn = 0.0
s_tmin = 0.0
s_tmax = 999.0
s_noise = 1.0

model = load_model("cpu", verbose=True).eval()
model_wrapped = SGMWrapper(model).to(device)
# model_wrapped = torch.compile(model_wrapped, dynamic=False)
ae = AutoEncoder(chunk_size=1).to(device)
conditioner = CLIPConditioner().to(device)
discretization = DDPMDiscretization()
guider = MultiviewCFG()
denoiser = DiscreteDenoiser(discretization=discretization, num_idx=1000, device=device)
sampler = EulerEDMSampler(
    discretization=discretization,
    guider=guider,
    num_steps=steps,
    s_churn=s_churn,
    s_tmin=s_tmin,
    s_tmax=s_tmax,
    s_noise=s_noise,
    verbose=True,
    device=device,
)

curr_imgs = data["curr_imgs"]
options = copy.deepcopy(data["options"])
H, W, T, C, F = curr_imgs.shape[-2], curr_imgs.shape[-1], data["T"], 4, 8
value_dict = get_value_dict(
    data["curr_imgs"],
    data["curr_imgs_clip"],
    data["curr_input_sels"],
    data["curr_c2ws"],
    data["curr_Ks"],
    list(range(T)),
    data["all_c2ws"],
    data["as_shuffled"],
    data["curr_input_frame_psuedo_indices"],
)
samples = do_sample(
    model_wrapped,
    ae,
    conditioner,
    denoiser,
    sampler,
    value_dict,
    H=H,
    W=W,
    C=C,
    F=F,
    T=T,
    cfg=3.0,
    encoding_t=1,
    decoding_t=1,
)
iio.imwrite(
    osp.join(work_dir, "stableviews_all.mp4"),
    (
        (samples.permute(0, 2, 3, 1) * 0.5 + 0.5).clamp(0, 1).cpu().numpy() * 255.0
    ).astype(np.uint8),
    fps=5,
)
__import__("ipdb").set_trace()
