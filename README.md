# Stable Virtual Camera: Generative View Synthesis with Diffusion Models

<a href=""><img src="https://img.shields.io/badge/%F0%9F%8F%A0
%20%20%20%20Project%20Page-gray.svg"></a> 
<a href="https://arxiv.org/abs/0000.0000"><img src="https://img.shields.io/badge/%F0%9F%93%84
%20%20%20%20arXiv-2408.00653-B31B1B.svg"></a> <a href=""><img src="https://img.shields.io/badge/%F0%9F%93%83
%20%20%20%20Blog-gray.svg"></a>  <a href="https://huggingface.co/stabilityai/stable-virtual-camera"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20%20%20Model_Card-Huggingface-orange"></a> <a href="https://huggingface.co/spaces/stabilityai/stable-virtual-camera"><img src="https://img.shields.io/badge/%F0%9F%9A%80%20%20%20%20Gradio%20Demo-Huggingface-orange"></a>

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

We provide two ways for you to interative with Seva: either via gradio demo or the command lines. 
- For general users, Gradio Demo is a pure GUI system that requires no expertised knowledge.
- For power users and academic researchers, Command-Line Demo equips you more powerful   fine-grained control of the model.

### Gradio Demo

For running the gradio demo, simply run 
```
python demo_gr.py
```

For a more detailed guide, follow [GR_GUIDE.md](GR_GUIDE.md)

### Command-Line Demo
For running the command-line script, refer to `demo.py` file and an examplar command line looks as simple as 
```
python demo.py --task img2img --data_path <data_path>
```

For a more detailed guide, follow [CLI_GUIDE.md](CLI_GUIDE.md)

# Citing
If you find this repository useful, please consider giving a star :star: and citation.
```
@article{zhou2025stable,
    title={Stable Virtual Camera: Generative View Synthesis with Diffusion Models},
    author={Jensen (Jinghao) Zhou and Hang Gao and Vikram Voleti and Aaryaman Vasishta and Chun-Han Yao and Mark Boss and
    Philip Torr and Christian Rupprecht and Varun Jampani
    },
    journal={arXiv preprint},
    year={2025}
}
```
