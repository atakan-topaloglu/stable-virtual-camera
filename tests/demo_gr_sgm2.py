import copy
import os.path as osp
import sys
from datetime import datetime

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import tyro
import viser
import viser.transforms as vt
from einops import rearrange

from scena.gui import define_gui

sys.path.insert(0, "/admin/home-hangg/projects/stable-research/")
from scripts.threeD_diffusion.demo.img2gui import Dust3rPipeline
from scripts.threeD_diffusion.run_eval import (
    infer_prior_stats,
    init_model,
    run_one_scene,
    transform_img_and_K,
)

device = "cuda:0"


work_dir = "/weka/home-hangg/projects/scena-release/work_dirs/_tests/demo_gr_sgm2"


class ScenaRender(object):
    def __init__(self, server: viser.ViserServer):
        self.server = server
        self.dust3r = Dust3rPipeline(device=device)  # type: ignore
        self.version_dict, self.engine = init_model(
            version="prediction_3D_SD21V_discrete_plucker_norm_replace",
            config="/admin/home-hangg/projects/stable-research/configs/3d_diffusion/jensen/inference/sd_3d-view-attn_21FT_discrete_no-clip-txt_pl---nk_plucker_concat_norm_mv_cat3d_v3_discrete_no-clip-txt_3d-attn-with-view-attn-mixing_freeze-pretrained_5355784_ckpt600000.yaml",
        )
        self.gui_state = None

    def preprocess(
        self,
        input_img_tuples: list[tuple[str, None]],
        shorter: int,
        use_center_crop: bool,
    ) -> dict:
        img_paths = [p for (p, _) in input_img_tuples]
        input_imgs, input_Ks, input_c2ws, points, point_colors = (
            self.dust3r.infer_cameras_and_points(img_paths)
        )
        num_inputs = len(img_paths)
        if num_inputs > 1:
            scene_scale = input_c2ws[:, :3, 3].ptp(-1).mean()
        else:
            scene_scale = 0.3
            input_imgs, input_Ks, input_c2ws, points, point_colors = (
                input_imgs[:1],
                input_Ks[:1],
                input_c2ws[:1],
                points[:1],
                point_colors[:1],
            )
        # Scale camera and points for viewport visualization.
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
            if use_center_crop:
                img, K = transform_img_and_K(img, (shorter, shorter), K=K[None])
            else:
                img, K = transform_img_and_K(img, shorter, K=K[None], size_stride=64)
            K = K / np.array([img.shape[-1], img.shape[-2], 1])[:, None]
            new_input_imgs.append(img)
            new_input_Ks.append(K)
        input_imgs = torch.cat(new_input_imgs, 0)
        input_imgs = rearrange(input_imgs, "b c h w -> b h w c")
        input_Ks = torch.cat(new_input_Ks, 0)
        if num_inputs > 1:
            scene_scale = input_c2ws[:, :3, 3].numpy().ptp(-1).mean()
        else:
            scene_scale = 0.3
            input_imgs, input_Ks, input_c2ws, points, point_colors = (
                input_imgs[:1],
                input_Ks[:1],
                input_c2ws[:1],
                points[:1],
                point_colors[:1],
            )
        return {
            "input_imgs": input_imgs,
            "input_Ks": input_Ks,
            "input_c2ws": input_c2ws,
            "input_wh": (input_imgs.shape[1], input_imgs.shape[2]),
            "points": points,
            "point_colors": point_colors,
            "scene_scale": scene_scale,
        }

    def visualize_scene(self, preprocessed: dict):
        server = self.server
        server.scene.reset()
        server.gui.reset()

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
        H, W = input_imgs[0].shape[:2]
        if H > W:
            init_fov = 2 * np.arctan(1 / (2 * input_Ks[0, 0, 0].item()))
        else:
            init_fov = 2 * np.arctan(1 / (2 * input_Ks[0, 1, 1].item()))
        init_fov_deg = float(init_fov / np.pi * 180.0)

        self.gui_state = define_gui(
            server,
            init_fov=init_fov_deg,
            img_wh=input_wh,
            scene_scale=scene_scale,
        )

        frustrums = []
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

            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frustum.wxyz
                    client.camera.position = frustum.position

            frustrums.append(frustum)

            server.scene.add_point_cloud(
                f"/scene_assets/points/{i}",
                points[i],
                point_colors[i],
                point_size=0.01 * scene_scale,
                point_shape="circle",
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
        render_dir = osp.join(work_dir, render_name)
        input_imgs, input_Ks, input_c2ws, input_wh = (
            preprocessed["input_imgs"],
            preprocessed["input_Ks"],
            preprocessed["input_c2ws"],
            preprocessed["input_wh"],
        )
        W, H = input_wh
        gui_state = self.gui_state
        assert gui_state is not None and gui_state.camera_path_list is not None
        target_c2ws, target_Ks = [], []
        for item in gui_state.camera_path_list:
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
        T = self.version_dict["T"]
        num_anchors, include_start_end = infer_prior_stats(
            T,
            num_inputs,
            num_total_frames=num_targets,
            options={"chunk_strategy": chunk_strategy},
        )
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
        options_ori = self.version_dict["options"]
        options = copy.deepcopy(options_ori)
        options["chunk_strategy"] = chunk_strategy
        options["video_save_fps"] = 30.0
        options["beta_linear_start"] = 5e-6
        options["log_snr_shift"] = 2.4
        options["cfg"] = (cfg, 2.0)
        options["use_traj_prior"] = True
        options["camera_scale"] = camera_scale
        for video_paths in run_one_scene(
            task="img2trajvid",
            version_dict={
                "H": H,
                "W": W,
                "T": T,
                "C": self.version_dict["C"],
                "f": self.version_dict["f"],
                "options": options,
            },
            model=self.engine,
            image_cond=image_cond,
            camera_cond=camera_cond,
            depths=None,
            save_path=render_dir,
            use_traj_prior=True,
            traj_prior_c2ws=anchor_c2ws,
            traj_prior_Ks=anchor_Ks,
            seed=seed,
            gradio=True,
        ):
            yield video_paths


_VIEWER_HEADER = """
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


def main(server_port: int | None = None):
    server = viser.ViserServer()
    server_url = server.request_share_url()
    renderer = ScenaRender(server)

    with gr.Blocks(head=_VIEWER_HEADER) as demo:
        show_viewport = gr.Checkbox(False, label="Show viewport", render=False)
        render_btn = gr.Button("Render video", interactive=False, render=False)
        gr.Timer(0.1).tick(
            lambda: gr.Button(
                interactive=renderer.gui_state is not None
                and renderer.gui_state.camera_path_list is not None
            ),
            outputs=[render_btn],
        )
        with gr.Row():
            viewport = gr.HTML(
                f'<iframe src="{server_url}" style="display: block; margin: auto; width: 100%; height: 60vh;" frameborder="0"></iframe>',
                container=True,
                visible=False,
            )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    preprocess_btn = gr.Button("Preprocess images")
                    show_viewport.render()
                    show_viewport.change(
                        lambda show: gr.HTML(visible=show),
                        inputs=[show_viewport],
                        outputs=[viewport],
                    )
                with gr.Row():
                    preprocessed = gr.State()
                    input_imgs = gr.Gallery(interactive=True, label="Input")
                    shorter = gr.Number(
                        value=576, label="Resize shorter side", render=False
                    )
                    use_center_crop = gr.Checkbox(
                        True,
                        label="Square crop? If not, keep aspect ratio",
                        render=False,
                    )
                    preprocess_btn.click(
                        renderer.preprocess,
                        inputs=[input_imgs, shorter, use_center_crop],
                        outputs=[preprocessed],
                    )
                    preprocess_btn.click(
                        lambda: gr.Checkbox(value=True),
                        outputs=[show_viewport],
                    )
                    preprocessed.change(
                        renderer.visualize_scene,
                        inputs=[preprocessed],
                        # show_progress_on=[show_viewport],
                    )
                with gr.Row():
                    shorter.render()
                    use_center_crop.render()
                with gr.Row():
                    seed = gr.Number(value=23, label="Random seed")
                    chunk_strategy = gr.Dropdown(
                        ["interp-gt", "interp"],
                        label="Chunk strategy",
                    )
                    cfg = gr.Slider(2.0, 10.0, value=3.0, label="CFG value")
                camera_scale = gr.Slider(
                    0.1,
                    10.0,
                    value=2.0,
                    label="Camera scale (useful for single image case)",
                )
            with gr.Column():
                render_btn.render()
                output_video_0 = gr.Video(
                    label="Intermediate output [1/2]", autoplay=True, loop=True
                )
                output_video_1 = gr.Video(
                    label="Final output [2/2]", autoplay=True, loop=True
                )
                render_btn.click(
                    renderer.render,
                    inputs=[
                        preprocessed,
                        seed,
                        chunk_strategy,
                        cfg,
                        camera_scale,
                    ],
                    outputs=[output_video_0, output_video_1],
                )

    try:
        demo.launch(
            share=True,
            server_port=server_port,
            allowed_paths=[work_dir],
        )
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    tyro.cli(main)
