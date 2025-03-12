# Stable Virtual Camera: Generative View Synthesis with Diffusion Models

<a href="https://arxiv.org/abs/0000.0000"><img src="https://img.shields.io/badge/Arxiv-2408.00653-B31B1B.svg"></a> <a href="https://huggingface.co/stabilityai/stable-virtual-camera"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a> <a href="https://huggingface.co/spaces/stabilityai/stable-virtual-camera"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a>


# [**Project Page**](https://stable-virtual-camera.github.io/)

`Stable Virtual Camera` is a 1.3 billion parameter multi-view diffusion model capable of generating 3D consistent novel views of a scene, given any number of input views and target cameras. For more information, please refer to our [paper](http://TODO) and [webpage](http://TODO).

# Installation
We recommend using running the setup.py script to setup the virtual environment and install all necessary dependencies on it:
`./setup.sh`

Or, if you want to install it manually:
```python
# For demo.
pip install -e .

# Or for development.
pip install -e ".[dev]"
pre-commit install

# [OPTIONAL] Install the torch nightly version for faster JIT via. torch.compile (speed up sampling by 2x in our testing).
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118
```

# Usage

For running the gradio demo, simply run 
```python demo_gr.py```

For running the CLI script:
```
python demo.py \
    --task img2img \
    --num_inputs 3 \ 
    --dataset_path <dataset_path> \
    --video_save_fps 10
```

For a more detailed guide, follow [CLI_GUIDE.md](CLI_GUIDE.md)