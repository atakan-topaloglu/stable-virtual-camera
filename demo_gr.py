import copy
import os
import os.path as osp
import secrets
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import httpx
import numpy as np
import torch
import torch.nn.functional as F
import tyro
import viser
import viser.transforms as vt
from einops import rearrange
from gradio import networking
from gradio.tunneling import CERTIFICATE_PATH, Tunnel

from stableviews.eval import (
    IS_TORCH_NIGHTLY,
    chunk_input_and_test,
    infer_prior_stats,
    run_one_scene,
    transform_img_and_K,
)
from stableviews.geometry import normalize_scene
from stableviews.gui import define_gui
from stableviews.model import SGMWrapper
from stableviews.modules.autoencoder import AutoEncoder
from stableviews.modules.conditioner import CLIPConditioner
from stableviews.modules.preprocessor import Dust3rPipeline
from stableviews.sampling import DDPMDiscretization, DiscreteDenoiser
from stableviews.utils import load_model

device = "cuda:0"


# Constants.
WORK_DIR = "work_dirs/demo_gr"
MAX_SESSIONS = 1
HEADER = """
<style>
  body,
  html {
    margin: 0;
    padding: 0;
    height: 100%;
  }
  iframe {
    border: 0;
    width: 100%;
    height: 100%;
  }
</style>
"""
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


class ScenaRenderer(object):
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
        # pbar = gr.Progress().tqdm(
        #     iterable=None,
        #     desc="Sampling",
        #     total=(num_chunks_0 + num_chunks_1) * num_steps,
        # )
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
            f"Maximum session ({MAX_SESSIONS}) reached. Please try again later. "
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
    return ScenaRenderer(server), gr.HTML(
        f'<iframe src="{server_url}" style="display: block; margin: auto; width: 100%; height: 60vh;" frameborder="0"></iframe>',
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
    with gr.Blocks(head=HEADER) as demo:
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
                        show_progress_on=[preprocess_progress],
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
                    show_progress_on=[render_progress],
                )
                render_btn.click(
                    lambda: gr.update(visible=True), outputs=[render_progress]
                )
        # Register the session initialization and cleanup functions.
        demo.load(start_server, outputs=[renderer, viewport])
        demo.unload(stop_server)

    demo.launch(
        share=share,
        server_port=server_port,
        show_error=True,
        allowed_paths=[WORK_DIR, EXAMPLE_DIR],
    )


if __name__ == "__main__":
    tyro.cli(main)
