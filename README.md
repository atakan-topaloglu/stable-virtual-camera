# Stable Virtual Camera

<a href=""><img src="https://img.shields.io/badge/%F0%9F%8F%A0
%20%20Project%20Page-gray.svg"></a> <a href="https://arxiv.org/abs/0000.0000"><img src="https://img.shields.io/badge/%F0%9F%93%84
%20%20arXiv-2408.00653-B31B1B.svg"></a> <a href=""><img src="https://img.shields.io/badge/%F0%9F%93%83
%20%20Blog-Stability%20AI-red.svg"></a>  <a href="https://huggingface.co/stabilityai/stable-virtual-camera"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a> <a href="https://huggingface.co/spaces/stabilityai/stable-virtual-camera"><img src="https://img.shields.io/badge/%F0%9F%9A%80%20%20Gradio%20Demo-Huggingface-orange"></a>

`Stable Virtual Camera (Seva)` is a generalist diffusion model for Novel View Synthesis (NVS), generating 3D consistent novel views of a scene, given any number of input views and target cameras. 


# :tada: News 
- March 2025 - `Seva` is out everywhere.

# Installation

To setup the virtual environment and install all necessary model dependencies, simply run:
```python
pip install -e .
```

Check [INSTALL_GUIDE.md](INSTALL_GUIDE.md) for other dependencies if you want to further use the provided demos.


# Demo Usage

We provide two demos for you to interative with `Seva`. 

### Gradio demo

This gradio demo is a GUI interface that requires no expertised knowledge, suitable for general users. Simply run 
```
python demo_gr.py
```

For a more detailed guide, follow [GR_GUIDE.md](GR_GUIDE.md)

### CLI demo
This cli demo allows you to pass in more options and control the model in a fine-grained way, suitable for power users and academic researchers. An examplar command line looks as simple as 
```
python demo.py --data_path <data_path> [additional arguments]
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
