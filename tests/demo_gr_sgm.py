import copy
import hashlib
import os
import os.path as osp
import sys
import time
from dataclasses import dataclass
from glob import glob
from typing import Literal

import gradio as gr
import imageio.v3 as iio
import numpy as np
import torch
import tyro
import viser
import viser.transforms as vt
from einops import rearrange, repeat
from tqdm import tqdm

sys.path.insert(0, "/admin/home-hangg/projects/stable-research/")
from scripts.threeD_diffusion.demo.img2gui import Dust3rPipeline, define_gui
from scripts.threeD_diffusion.run_eval import (
    DEFAULT_FOV_RAD,
    CovariancePredictionGuider,
    LinearPredictionGuider,
    LinearTrianglePredictionGuider,
    MultiPredictionGuider,
    MultiTrianglePredictionGuider,
    PostHocDecoderWithTime,
    TrapezoidPredictionGuider,
    VanillaCFG,
    VideoPredictionEmbedderWithEncoder,
    chunk_input_and_test,
    decode_output,
    default,
    extend_dict,
    generate_spiral_path,
    get_arc_horizontal_w2cs,
    get_default_intrinsics,
    get_k_from_dict,
    get_lemniscate_w2cs,
    get_panning_w2cs,
    get_plucker_coordinates,
    get_roll_w2cs,
    get_unique_embedder_keys_from_conditioner,
    infer_prior_stats,
    init_embedder_options_no_st,
    init_model,
    init_sampling_no_st,
    load_model,
    seed_everything,
    to_hom_pose,
    transform_img_and_K,
    unload_model,
    update_kv_for_dict,
)

device = "cuda:0"


@dataclass
class Task(object):
    input_imgs: torch.Tensor
    input_c2ws: torch.Tensor
    input_Ks: torch.Tensor
    target_c2ws: torch.Tensor
    target_Ks: torch.Tensor
    num_inputs: int
    num_targets: int
    input_indices: list[int]
    target_indices: list[int]
    camera_scale: float = 2.0


@dataclass
class ChunkedTask(object):
    input_imgs: torch.Tensor
    input_c2ws: torch.Tensor
    input_Ks: torch.Tensor
    target_c2ws: torch.Tensor
    target_Ks: torch.Tensor
    num_inputs: int
    num_targets: int
    input_indices: list[int]
    target_indices: list[int]
    camera_scale: float

    anchor_c2ws: torch.Tensor
    anchor_Ks: torch.Tensor
    num_anchors: int
    anchor_indices: list[int]


@dataclass
class ScenaRenderOptions(object):
    seed: int


@dataclass
class ScenaRendererConfig:
    context_window: int = 21
    output_root = "work_dirs/_tests/demo_gr_sgm/"
    chunk_strategy: Literal["nearest", "interp", "interp-gt"] = "interp-gt"
    chunk_strategy_first_pass: str = "gt_nearest"
    seed: int = 23

    def __post_init__(self):
        self.options = {
            "as_shuffled": True,
            "discretization": 0,
            "beta_linear_start": 5e-06,
            "beta_linear_end": 0.012,
            "beta_schedule": "linear",
            "log_snr_shift": 2.4,
            "guider": (6, 5),
            "cfg": (3.0, 2.0),
            "cfg_min": 1.2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 50,
            "additional_guider_kwargs": {
                "additional_cond_keys": ["replace", "dense_vector"],
                "scale_rule_config": {
                    "target": "sgm.modules.diffusionmodules.guiders.LowCFGCloseInputFrame",
                    "params": {"scale_low": 1.2},
                },
                "scale_schedule_config": {
                    "target": "sgm.modules.diffusionmodules.guiders.IdentitySchedule",
                    "params": {"scale": 2.0},
                },
                "dyn_thresh_config": {
                    "target": "sgm.modules.diffusionmodules.guiders.NoDynamicThresholding"
                },
            },
            "num_frames": self.context_window,
        }


class ScenaRenderer(object):
    def __init__(self, cfg: ScenaRendererConfig, server: viser.ViserServer):
        self.cfg = cfg
        self.dust3r_processor = Dust3rPipeline(device=device)  # type: ignore
        self.server = server

        _, self.engine = init_model(
            version="prediction_3D_SD21V_discrete_plucker_norm_replace",
            config="/admin/home-hangg/projects/stable-research/configs/3d_diffusion/jensen/inference/sd_3d-view-attn_21FT_discrete_no-clip-txt_pl---nk_plucker_concat_norm_mv_cat3d_v3_discrete_no-clip-txt_3d-attn-with-view-attn-mixing_freeze-pretrained_5355784_ckpt600000.yaml",
        )
        self.embedder_keys = set(
            get_unique_embedder_keys_from_conditioner(self.engine.conditioner)
        )

    def preprocess(
        self,
        input_img_tuples: list[tuple[str, None]],
        shorter: int,
        use_center_crop: bool,
    ) -> Task:
        img_paths = [p for (p, _) in input_img_tuples]
        input_imgs, input_Ks, input_c2ws, points, point_colors = (
            self.dust3r_processor.infer_cameras_and_points(img_paths)
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
        # scale camera and points for GUI visualization
        input_c2ws[:, :3, 3] /= scene_scale
        points = [point / scene_scale for point in points]

        input_imgs = [
            torch.as_tensor(img / 255.0, dtype=torch.float32) for img in input_imgs
        ]
        input_Ks = torch.as_tensor(input_Ks)
        input_c2ws = torch.as_tensor(input_c2ws)
        new_input_imgs, new_input_Ks = [], []
        for img, K in zip(input_imgs, input_Ks):
            img = rearrange(img, "h w c -> 1 c h w")
            if use_center_crop:
                img, K = transform_img_and_K(img, (shorter, shorter), K=K[None])
            else:
                assert shorter % 64 == 0
                img, K = transform_img_and_K(img, shorter, K=K[None], size_stride=64)
            K = K / np.array([img.shape[-1], img.shape[-2], 1])[:, None]
            new_input_imgs.append(img)
            new_input_Ks.append(K)
        input_imgs = torch.cat(new_input_imgs, 0)
        input_imgs = rearrange(input_imgs, "b c h w -> b h w c")
        input_Ks = torch.cat(new_input_Ks, 0)

        # TODO(hangg): Sometime duster will give fx != fy, it will break the
        # viser fov logic.
        H, W = input_imgs[0].shape[:2]  # first image as default fov
        if H > W:
            init_fov = 2 * np.arctan(1 / (2 * input_Ks[0, 0, 0]))
        else:
            init_fov = 2 * np.arctan(1 / (2 * input_Ks[0, 1, 1]))

        server = self.server
        server.scene.reset()
        server.scene.set_up_direction(-input_c2ws[..., :3, 1].mean(0).numpy())
        gui_state = define_gui(
            server,
            "/tmp/_scena_tmp_camera_paths",
            init_fov=float(init_fov / np.pi * 180),
            img_wh=(W, H),
            scene_scale=scene_scale,
        )

        frustrums = []
        for i in range(len(input_imgs)):
            K = input_Ks[i]
            H, W = input_imgs[i].shape[:2]
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

        switch_button = server.gui.add_button(
            "Switch Input Camera",
            icon=viser.Icon.EXCHANGE,
            hint="Switch to another input pose.",
        )

        @switch_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client_id is not None
            client = server.get_clients()[event.client_id]
            client.camera.wxyz = frustrums[gui_state.active_input_index].wxyz
            client.camera.position = frustrums[gui_state.active_input_index].position
            gui_state.active_input_index = (gui_state.active_input_index + 1) % len(
                frustrums
            )

        def get_target_cameras():
            camera_path_list = gui_state.camera_path_list
            assert camera_path_list is not None
            target_c2ws, target_Ks = [], []
            for item in camera_path_list:
                target_c2ws.append(item["w2c"])
                W, H = item["img_wh"]
                K = np.array(item["K"]).reshape(3, 3) / np.array([W, H, 1])[:, None]
                target_Ks.append(K)
            target_c2ws = np.linalg.inv(np.array(target_c2ws).reshape(-1, 4, 4))
            target_Ks = np.array(target_Ks).reshape(-1, 3, 3)
            return target_c2ws, target_Ks

        while gui_state.camera_path_list is None:
            time.sleep(0.1)
        target_c2ws, target_Ks = get_target_cameras()
        target_Ks = torch.as_tensor(target_Ks)
        target_c2ws = torch.as_tensor(target_c2ws)

        input_imgs = rearrange(input_imgs, "b h w c -> b c h w")
        num_inputs = len(input_imgs)
        num_targets = len(target_c2ws)
        input_indices = list(range(num_inputs))
        target_indices = np.arange(num_inputs, num_inputs + num_targets).tolist()
        return Task(
            input_imgs,
            input_c2ws,
            input_Ks,
            target_c2ws,
            target_Ks,
            num_inputs,
            num_targets,
            input_indices,
            target_indices,
        )

    def chunk(self, task: Task) -> ChunkedTask:
        cfg = self.cfg
        num_anchors, include_start_end = infer_prior_stats(
            cfg.context_window,
            task.num_inputs,
            num_total_frames=task.num_targets,
            options={"chunk_strategy": cfg.chunk_strategy},
        )
        assert isinstance(num_anchors, int)
        anchor_c2ws = task.target_c2ws[
            torch.linspace(0, task.num_targets - 1, num_anchors).round().long()
        ]
        anchor_Ks = repeat(task.target_Ks[:1], "1 i j -> b i j", b=num_anchors)
        anchor_indices = np.linspace(
            task.num_inputs,
            task.num_inputs + task.num_targets - 1,
            num_anchors + 1 - include_start_end,
            endpoint=include_start_end,
        )[1 - include_start_end :].tolist()
        return ChunkedTask(
            task.input_imgs,
            task.input_c2ws,
            task.input_Ks,
            task.target_c2ws,
            task.target_Ks,
            task.num_inputs,
            task.num_targets,
            task.input_indices,
            task.target_indices,
            task.camera_scale,
            anchor_c2ws,
            anchor_Ks,
            num_anchors,
            anchor_indices,
        )

    def render_video(self, task: Task, options: ScenaRenderOptions):
        print(options.seed)
        chunked_task = self.chunk(task)
        data_name = hashlib.sha256(task.input_imgs[0].numpy().tobytes()).hexdigest()[
            :16
        ]
        W, H = task.input_imgs.shape[-1], task.input_imgs.shape[-2]

        output_dir = osp.join(self.cfg.output_root, data_name, f"{task.num_targets}")

        input_samples = rearrange(
            chunked_task.input_imgs.cpu().numpy() * 255.0,
            "n c h w -> n h w c",
        ).astype(np.uint8)
        input_dir = osp.join(output_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        for i, img in enumerate(input_samples):
            iio.imwrite(osp.join(input_dir, f"{i:04d}.png"), img)
        input_path = osp.join(output_dir, "input.mp4")
        iio.imwrite(input_path, input_samples, fps=2.0)

        samplers = init_sampling_no_st(options=self.cfg.options)

        seed_everything(options.seed)

        (
            _,
            input_inds_per_chunk,
            input_sels_per_chunk,
            anchor_inds_per_chunk,
            anchor_sels_per_chunk,
        ) = chunk_input_and_test(
            self.cfg.context_window,
            chunked_task.input_c2ws,
            chunked_task.anchor_c2ws,
            chunked_task.input_indices,
            chunked_task.anchor_indices,
            task="img2trajvid",
            chunk_strategy=self.cfg.chunk_strategy_first_pass,
            gt_input_inds=list(range(chunked_task.num_inputs)),
            verbose=True,
        )
        print(
            f"Two passes (first) - chunking with `{self.cfg.chunk_strategy_first_pass}` strategy: total "
            f"{len(input_inds_per_chunk)} forward(s) ..."
        )

        all_samples = {}
        all_anchor_inds = []
        for i, (
            chunk_input_inds,
            chunk_input_sels,
            chunk_anchor_inds,
            chunk_anchor_sels,
        ) in tqdm(
            enumerate(
                zip(
                    input_inds_per_chunk,
                    input_sels_per_chunk,
                    anchor_inds_per_chunk,
                    anchor_sels_per_chunk,
                )
            ),
            total=len(input_inds_per_chunk),
            leave=False,
        ):
            (
                curr_input_sels,
                curr_anchor_sels,
                curr_input_maps,
                curr_anchor_maps,
            ) = self.pad_indices(
                chunk_input_sels,
                chunk_anchor_sels,
                padding_mode="last",
            )
            curr_imgs, curr_imgs_clip, curr_c2ws, curr_Ks, curr_depths = [
                self.assemble(
                    input=x[chunk_input_inds],
                    test=y[chunk_anchor_inds],
                    input_maps=curr_input_maps,
                    test_maps=curr_anchor_maps,
                )
                for x, y in zip(
                    [
                        torch.cat(
                            [
                                chunked_task.input_imgs * 2.0 - 1.0,
                                get_k_from_dict(all_samples, "samples-rgb").to(
                                    chunked_task.input_imgs.device
                                ),
                            ],
                            dim=0,
                        ),
                        torch.cat(
                            [
                                chunked_task.input_imgs * 2.0 - 1.0,
                                get_k_from_dict(all_samples, "samples-rgb").to(
                                    chunked_task.input_imgs.device
                                ),
                            ],
                            dim=0,
                        ),
                        torch.cat(
                            [
                                chunked_task.input_c2ws,
                                chunked_task.anchor_c2ws[all_anchor_inds],
                            ],
                            dim=0,
                        ),
                        torch.cat(
                            [
                                chunked_task.input_Ks,
                                chunked_task.anchor_Ks[all_anchor_inds],
                            ],
                            dim=0,
                        ),
                        torch.cat(
                            [
                                chunked_task.input_Ks,
                                chunked_task.anchor_Ks[all_anchor_inds],
                            ],
                            dim=0,
                        ),
                    ],  # procedually append generated prior views to the input views
                    [
                        repeat(
                            torch.zeros_like(chunked_task.input_imgs[:1]),
                            "1 c h w -> n c h w",
                            n=chunked_task.num_anchors,
                        ),
                        repeat(
                            torch.zeros_like(chunked_task.input_imgs[:1]),
                            "1 c h w -> n c h w",
                            n=chunked_task.num_anchors,
                        ),
                        chunked_task.anchor_c2ws,
                        chunked_task.anchor_Ks,
                        chunked_task.anchor_Ks,
                    ],
                )
            ]
            value_dict = self.get_value_dict(
                curr_imgs,
                curr_imgs_clip,
                curr_input_sels,
                curr_c2ws,
                curr_Ks,
                list(range(self.cfg.context_window)),
                curr_depths,
                all_c2ws=torch.cat(
                    [chunked_task.input_c2ws, chunked_task.target_c2ws], 0
                ),
                as_shuffled=(
                    self.cfg.options["as_shuffled"]
                    and any(
                        map(
                            lambda x: x in self.cfg.chunk_strategy_first_pass,
                            ["nearest", "gt"],
                        )
                    )  # "interp" not in chunk_strategy_first_pass
                ),
                curr_input_frame_psuedo_indices=[
                    sel
                    for ind, sel in zip(chunk_input_inds, curr_input_sels)
                    if ind >= len(chunked_task.input_indices)
                ],
                camera_scale=chunked_task.camera_scale,
            )
            # value_dict = torch.load(
            #     "/admin/home-hangg/projects/scena-release/tests/demo_gr_value_dict.pth"
            # )

            samples = self.do_sample(
                (
                    samplers[1]
                    if len(samplers) > 1
                    and self.cfg.options.get("ltr_first_pass", False)
                    and self.cfg.chunk_strategy_first_pass != "gt"
                    and i > 0
                    else samplers[0]
                ),
                value_dict,
                H,
                W,
                4,
                8,
                T=self.cfg.context_window,
                batch2model_input=["num_video_frames", "image_only_indicator"],
                **self.cfg.options,
            )
            samples = decode_output(samples, self.cfg.context_window, chunk_anchor_sels)
            extend_dict(all_samples, samples)
            all_anchor_inds.extend(chunk_anchor_inds)

        first_pass_samples = rearrange(
            (all_samples["samples-rgb/image"] / 2.0 + 0.5).clamp(0.0, 1.0).cpu().numpy()
            * 255.0,
            "n c h w -> n h w c",
        ).astype(np.uint8)
        first_pass_dir = osp.join(output_dir, "first_pass")
        os.makedirs(first_pass_dir, exist_ok=True)
        for i, img in enumerate(first_pass_samples):
            iio.imwrite(osp.join(first_pass_dir, f"{i:04d}.png"), img)
        first_pass_path = osp.join(output_dir, "first_pass.mp4")
        iio.imwrite(first_pass_path, first_pass_samples, fps=5.0)
        yield first_pass_path, None

        assert (
            chunked_task.anchor_indices is not None
        ), "`anchor_frame_indices` needs to be set if using 2-pass sampling."
        anchor_argsort = np.argsort(
            np.array(chunked_task.input_indices + chunked_task.anchor_indices)
        ).tolist()
        anchor_indices = np.array(
            chunked_task.input_indices + chunked_task.anchor_indices
        )[anchor_argsort].tolist()
        gt_input_inds = [
            anchor_argsort.index(i) for i in range(chunked_task.input_c2ws.shape[0])
        ]

        anchor_imgs = torch.cat(
            [
                chunked_task.input_imgs,
                get_k_from_dict(all_samples, "samples-rgb") / 2.0 + 0.5,
            ],
            dim=0,
        )[anchor_argsort]
        anchor_c2ws = torch.cat(
            [chunked_task.input_c2ws, chunked_task.anchor_c2ws], dim=0
        )[anchor_argsort]
        anchor_Ks = torch.cat([chunked_task.input_Ks, chunked_task.anchor_Ks], dim=0)[
            anchor_argsort
        ]

        update_kv_for_dict(all_samples, "samples-rgb", anchor_imgs)
        update_kv_for_dict(all_samples, "samples-c2ws", anchor_c2ws)
        update_kv_for_dict(all_samples, "samples-intrinsics", anchor_Ks)

        (
            _,
            anchor_inds_per_chunk,
            anchor_sels_per_chunk,
            target_inds_per_chunk,
            target_sels_per_chunk,
        ) = chunk_input_and_test(
            self.cfg.context_window,
            anchor_c2ws,
            chunked_task.target_c2ws,
            anchor_indices,
            chunked_task.target_indices,
            task="img2trajvid",
            chunk_strategy=self.cfg.chunk_strategy,
            gt_input_inds=gt_input_inds,
            verbose=self.cfg.options.get("sampler_verbose", True),
        )
        print(
            f"Two passes (second) - chunking with `{self.cfg.chunk_strategy}` strategy: total "
            f"{len(anchor_inds_per_chunk)} forward(s) ..."
        )

        all_samples = {}
        all_target_inds = []
        for i, (
            chunk_anchor_inds,
            chunk_anchor_sels,
            chunk_target_inds,
            chunk_target_sels,
        ) in tqdm(
            enumerate(
                zip(
                    anchor_inds_per_chunk,
                    anchor_sels_per_chunk,
                    target_inds_per_chunk,
                    target_sels_per_chunk,
                )
            ),
            total=len(anchor_inds_per_chunk),
            leave=False,
        ):
            (
                curr_anchor_sels,
                curr_target_sels,
                curr_anchor_maps,
                curr_target_maps,
            ) = self.pad_indices(
                chunk_anchor_sels,
                chunk_target_sels,
                padding_mode="last",
            )
            curr_imgs, curr_imgs_clip, curr_c2ws, curr_Ks, curr_depths = [
                self.assemble(
                    input=x[chunk_anchor_inds],
                    test=y[chunk_target_inds],
                    input_maps=curr_anchor_maps,
                    test_maps=curr_target_maps,
                )
                for x, y in zip(
                    [
                        anchor_imgs * 2.0 - 1.0,
                        anchor_imgs * 2.0 - 1.0,
                        anchor_c2ws,
                        anchor_Ks,
                        anchor_Ks,
                    ],
                    [
                        repeat(
                            torch.zeros_like(chunked_task.input_imgs[:1]),
                            "1 c h w -> n c h w",
                            n=chunked_task.num_targets,
                        ),
                        repeat(
                            torch.zeros_like(chunked_task.input_imgs[:1]),
                            "1 c h w -> n c h w",
                            n=chunked_task.num_targets,
                        ),
                        chunked_task.target_c2ws,
                        chunked_task.target_Ks,
                        chunked_task.target_Ks,
                    ],
                )
            ]
            value_dict = self.get_value_dict(
                curr_imgs,
                curr_imgs_clip,
                curr_anchor_sels,
                curr_c2ws,
                curr_Ks,
                list(range(self.cfg.context_window)),
                curr_depths,
                all_c2ws=torch.cat(
                    [chunked_task.input_c2ws, chunked_task.target_c2ws], 0
                ),
                as_shuffled=(
                    self.cfg.options["as_shuffled"]
                    and any(
                        map(
                            lambda x: x in self.cfg.chunk_strategy,
                            ["nearest", "gt"],
                        )
                    )  # "interp" not in chunk_strategy
                ),
                camera_scale=chunked_task.camera_scale,
            )
            samples = self.do_sample(
                samplers[1] if len(samplers) > 1 else samplers[0],
                value_dict,
                H,
                W,
                4,
                8,
                T=self.cfg.context_window,
                batch2model_input=["num_video_frames", "image_only_indicator"],
                **self.cfg.options,
            )
            samples = decode_output(samples, self.cfg.context_window, chunk_target_sels)
            extend_dict(all_samples, samples)
            all_target_inds.extend(chunk_target_inds)

        all_samples = {
            key: value[np.argsort(all_target_inds)]
            for key, value in all_samples.items()
        }
        second_pass_samples = rearrange(
            (all_samples["samples-rgb/image"] / 2.0 + 0.5).clamp(0.0, 1.0).cpu().numpy()
            * 255.0,
            "n c h w -> n h w c",
        ).astype(np.uint8)
        second_pass_dir = osp.join(output_dir, "second_pass")
        os.makedirs(second_pass_dir, exist_ok=True)
        for i, img in enumerate(second_pass_samples):
            iio.imwrite(osp.join(second_pass_dir, f"{i:04d}.png"), img)
        second_pass_path = osp.join(output_dir, "second_pass.mp4")
        iio.imwrite(second_pass_path, second_pass_samples, fps=30.0)
        yield first_pass_path, second_pass_path

    def get_value_dict(
        self,
        curr_imgs,
        curr_imgs_clip,
        curr_input_frame_indices,
        curr_c2ws,
        curr_Ks,
        curr_input_camera_indices,
        curr_depths,
        all_c2ws,
        camera_scale=2.0,
        as_shuffled=True,
        curr_input_frame_psuedo_indices=[],
    ):
        assert sorted(curr_input_camera_indices) == sorted(
            range(len(curr_input_camera_indices))
        )
        H, W, T, F = curr_imgs.shape[-2], curr_imgs.shape[-1], len(curr_imgs), 8

        value_dict = init_embedder_options_no_st(
            self.embedder_keys,
            prompt="A scene.",
            negative_prompt="",
        )
        value_dict["image_only_indicator"] = int(as_shuffled)

        value_dict["cond_frames_without_noise"] = curr_imgs_clip[
            curr_input_frame_indices
        ]
        T = len(curr_imgs)
        value_dict["cond_frames_mask"] = torch.zeros(T, dtype=torch.bool)
        value_dict["cond_frames_mask"][curr_input_frame_indices] = True
        value_dict["cond_frames_psuedo_mask"] = torch.zeros(T, dtype=torch.bool)
        value_dict["cond_frames_psuedo_mask"][curr_input_frame_psuedo_indices] = True
        # This is only used for matching random seeding with `run_eval.py`.
        value_dict["cond_frames"] = curr_imgs + 0.0 * torch.randn_like(curr_imgs)
        value_dict["cond_aug"] = 0.0

        c2w = to_hom_pose(curr_c2ws.float())
        w2c = torch.linalg.inv(c2w)

        # camera centering
        ref_c2ws = all_c2ws
        camera_dist_2med = torch.norm(
            ref_c2ws[:, :3, 3] - ref_c2ws[:, :3, 3].median(0, keepdim=True).values,
            dim=-1,
        )
        valid_mask = camera_dist_2med <= torch.clamp(
            torch.quantile(camera_dist_2med, 0.97) * 10,
            max=1e6,
        )
        c2w[:, :3, 3] -= ref_c2ws[valid_mask, :3, 3].mean(0, keepdim=True)
        w2c = torch.linalg.inv(c2w)

        # camera normalization
        camera_dists = c2w[:, :3, 3].clone()
        translation_scaling_factor = (
            camera_scale
            if torch.isclose(
                torch.norm(camera_dists[0]), torch.zeros(1), atol=1e-6
            ).any()
            else (camera_scale / torch.norm(camera_dists[0]))
        )
        w2c[:, :3, 3] *= translation_scaling_factor
        c2w[:, :3, 3] *= translation_scaling_factor
        value_dict["plucker_coordinate"] = get_plucker_coordinates(
            extrinsics_src=w2c[0],
            extrinsics=w2c,
            intrinsics=curr_Ks.float().clone(),
            mode="plucker",
            rel_zero_translation=True,
            target_size=(H // F, W // F),
            return_grid_cam=True,
        )[0]

        value_dict["c2w"] = c2w  # used by guider
        value_dict["K"] = curr_Ks  # used by guider
        value_dict["camera_mask"] = torch.zeros(T, dtype=torch.bool)
        value_dict["camera_mask"][curr_input_camera_indices] = True

        return value_dict

    def pad_indices(
        self,
        input_indices: list[int],
        test_indices: list[int],
        padding_mode: Literal["first", "last", "none"] = "last",
    ):
        T = self.cfg.context_window
        assert padding_mode in ["last", "none"], "`first` padding is not supported yet."
        if padding_mode == "last":
            padded_indices = [
                i for i in range(T) if i not in (input_indices + test_indices)
            ]
        else:
            padded_indices = []
        input_selects = list(range(len(input_indices)))
        test_selects = list(range(len(test_indices)))
        if max(input_indices) > max(test_indices):
            # last elem from input
            input_selects += [input_selects[-1]] * len(padded_indices)
            input_indices = input_indices + padded_indices
            sorted_inds = np.argsort(np.array(input_indices))
            input_indices = [input_indices[ind] for ind in sorted_inds]
            input_selects = [input_selects[ind] for ind in sorted_inds]
        else:
            # last elem from test
            test_selects += [test_selects[-1]] * len(padded_indices)
            test_indices = test_indices + padded_indices
            sorted_inds = np.argsort(test_indices)
            test_indices = [test_indices[ind] for ind in sorted_inds]
            test_selects = [test_selects[ind] for ind in sorted_inds]

        if padding_mode == "last":
            input_maps = np.array([-1] * T)
            test_maps = np.array([-1] * T)
        else:
            input_maps = np.array([-1] * (len(input_indices) + len(test_indices)))
            test_maps = np.array([-1] * (len(input_indices) + len(test_indices)))
        input_maps[input_indices] = input_selects
        test_maps[test_indices] = test_selects
        return input_indices, test_indices, input_maps, test_maps

    def assemble(
        self,
        input,
        test,
        input_maps,
        test_maps,
    ):
        # Support padding for legacy reason, the right way is to do the
        # attention masking.
        T = len(input_maps)
        assembled = torch.zeros_like(test[-1:]).repeat_interleave(T, dim=0)
        assembled[input_maps != -1] = input[input_maps[input_maps != -1]]
        assembled[test_maps != -1] = test[test_maps[test_maps != -1]]
        assert np.logical_xor(input_maps != -1, test_maps != -1).all()
        return assembled

    def get_batch(
        self,
        keys,
        value_dict: dict,
        N: list[int],
        device: str = "cuda",
        T: int = 21,
        additional_batch_uc_fields: list[str] = [],
    ):
        # Hardcoded demo setups; might undergo some changes in the future
        batch = {}
        batch_uc = {}
        prod_N = np.prod(np.array(N)).item()

        for key in keys:
            if key == "txt":
                batch["txt"] = [value_dict["prompt"]] * prod_N
                batch_uc["txt"] = [value_dict["negative_prompt"]] * prod_N
            elif key == "cond_aug":
                batch[key] = repeat(
                    torch.tensor([value_dict["cond_aug"]]).to(device),
                    "1 -> b",
                    b=prod_N,
                )
            elif (
                key == "cond_frames"
                or key == "cond_frames_mask"
                or key == "cond_frames_without_noise"
            ):
                value = value_dict[key]
                batch[key] = repeat(
                    value if isinstance(value, torch.Tensor) else torch.tensor(value),
                    "n ... -> (b n) ...",
                    b=prod_N // T,
                ).to(device)
            elif (
                key == "polar_rad"
                or key == "azimuth_rad"
                or key == "plucker_coordinate"
                or key == "camera_mask"
            ):
                value = value_dict[key]
                batch[key] = repeat(
                    value if isinstance(value, torch.Tensor) else torch.tensor(value),
                    "n ... -> (b n) ...",
                    b=prod_N // T,
                ).to(device)
                # for logging as gt
                if key == "plucker_coordinate":
                    (
                        batch["plucker_coordinate_direction"],
                        batch["plucker_coordinate_moment"],
                    ) = batch["plucker_coordinate"].chunk(2, dim=1)
            else:
                batch[key] = value_dict[key]

        # for logging as gt
        for key in ["depth_lowres", "depth_lowres/raw", "point"]:
            if key in value_dict:
                batch[key] = repeat(
                    value_dict[key],
                    "n ... -> (b n) ...",
                    b=prod_N // T,
                ).to(device)

        if T is not None:
            batch["num_video_frames"] = T

        for key in batch.keys():
            if key not in batch_uc and isinstance(batch[key], torch.Tensor):
                batch_uc[key] = torch.clone(batch[key])
            elif key in additional_batch_uc_fields and key not in batch_uc:
                batch_uc[key] = copy.copy(batch[key])
        return batch, batch_uc

    def do_sample(
        self,
        sampler_,
        value_dict,
        H,
        W,
        C,
        F,
        num_samples: int = 1,
        force_uc_zero_embeddings: list[str] = [],
        force_cond_zero_embeddings: list[str] = [],
        batch2model_input: list[str] = [],
        T=None,
        additional_batch_uc_fields: list[str] = [],
        encoding_t=None,
        decoding_t=None,
        log_groundtruth=False,
        keys_to_log_from_gt=(),
        log_gt_frames=False,
        log_processed=False,
        dataset_latent=False,
        **_,
    ):
        model = self.engine
        if encoding_t:
            for emb in model.conditioner.embedders:
                if isinstance(emb, VideoPredictionEmbedderWithEncoder):
                    emb.en_and_decode_n_samples_a_time = encoding_t
        if decoding_t:
            model.en_and_decode_n_samples_a_time = decoding_t

        precision_scope = torch.autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    if T is not None:
                        num_samples = [num_samples, T]
                    else:
                        num_samples = [num_samples]

                    load_model(model.conditioner)
                    batch, batch_uc = self.get_batch(
                        get_unique_embedder_keys_from_conditioner(model.conditioner),
                        value_dict,
                        num_samples,
                        T=T,
                        additional_batch_uc_fields=additional_batch_uc_fields,
                    )

                    c, uc = model.conditioner.get_unconditional_conditioning(
                        batch,
                        batch_uc=batch_uc,
                        force_uc_zero_embeddings=force_uc_zero_embeddings,
                        force_cond_zero_embeddings=force_cond_zero_embeddings,
                        n_cond_frames=T,
                    )
                    unload_model(model.conditioner)

                    for k in c:
                        if not k == "crossattn":
                            c[k], uc[k] = map(
                                lambda y: y[k][: np.prod(num_samples).item()].to(
                                    "cuda"
                                ),
                                (c, uc),
                            )

                    additional_model_inputs = {}
                    for k in batch2model_input:
                        if k == "image_only_indicator":
                            assert T is not None

                            if isinstance(
                                sampler_.guider,
                                (
                                    VanillaCFG,
                                    LinearPredictionGuider,
                                    LinearTrianglePredictionGuider,
                                    TrapezoidPredictionGuider,
                                    MultiTrianglePredictionGuider,
                                    MultiPredictionGuider,
                                    CovariancePredictionGuider,
                                ),
                            ):
                                additional_model_inputs[k] = torch.full(
                                    (num_samples[0] * 2, num_samples[1]),
                                    value_dict[k],
                                ).to("cuda")
                            else:
                                additional_model_inputs[k] = (
                                    value_dict[k].repeat(num_samples).to("cuda")
                                )
                        else:
                            additional_model_inputs[k] = batch[k]

                    shape = (np.prod(num_samples).item(), C, H // F, W // F)
                    randn = torch.randn(shape).to("cuda")

                    #####
                    def denoiser(input, sigma, c):
                        return model.denoiser(
                            model.model, input, sigma, c, **additional_model_inputs
                        )

                    # THIS IS VERY IMPORTANT! IF your denoiser is discrete denoiser
                    # that requries discretization, you should make sure the one used
                    # is the same one used for sampler. This is to avoid potential
                    # inconsistency between the training and inference config.
                    if hasattr(model.denoiser, "discretization"):
                        model.denoiser.discretization = sampler_.discretization
                        model.denoiser.register_sigmas()
                    #######

                    additional_sampler_inputs = {
                        "c2w": value_dict["c2w"].to("cuda"),
                        "K": value_dict["K"].to("cuda"),
                        "input_frame_mask": value_dict["cond_frames_mask"].to("cuda"),
                        "input_frame_psuedo_mask": value_dict[
                            "cond_frames_psuedo_mask"
                        ].to("cuda"),
                    }

                    load_model(model.denoiser)
                    load_model(model.model)
                    samples_z = sampler_(
                        denoiser,
                        randn,
                        cond=c,
                        uc=uc,
                        **additional_sampler_inputs,
                    )
                    # sampler_input_dict = torch.load(
                    #     "/admin/home-hangg/projects/scena-release/tests/demo_gr_sampler_input_dict.pth"
                    # )
                    # __import__("ipdb").set_trace()
                    # samples_z = sampler_(
                    #     denoiser,
                    #     sampler_input_dict["randn"],
                    #     cond=sampler_input_dict["c"],
                    #     uc=sampler_input_dict["uc"],
                    #     **sampler_input_dict["additional_sampler_inputs"],
                    # )
                    # __import__("ipdb").set_trace()
                    unload_model(model.model)
                    unload_model(model.denoiser)

                    load_model(model.first_stage_model)
                    if isinstance(
                        model.first_stage_model.decoder, PostHocDecoderWithTime
                    ):
                        samples = model.decode_first_stage(
                            samples_z, timesteps=default(decoding_t, T)
                        )
                    else:
                        samples = model.decode_first_stage(samples_z)

                    if isinstance(samples, dict) and model.postprocessor_for_logging:
                        if log_groundtruth:
                            samples_new = model.postprocessor_for_logging(
                                batch,
                                {"samples": samples},
                                n=np.prod(num_samples).item(),
                                log_from_batch=True,
                                log_raw=True,
                                log_processed=log_processed,
                            )
                            for k in (
                                keys_to_log_from_gt
                                if isinstance(keys_to_log_from_gt, (tuple, list))
                                else keys_to_log_from_gt.split(",")
                            ):
                                if k in batch:
                                    samples_new[k] = batch[k]
                        else:
                            samples.update(
                                model.postprocessor_for_logging(
                                    {"num_video_frames": T, **samples},
                                    n=np.prod(num_samples).item(),
                                    log_from_batch=True,
                                    log_raw=True,
                                    log_processed=log_processed,
                                )
                            )
                            samples_new = {}

                        for k, v in samples.items():
                            samples_new[f"samples-{k}"] = v
                        samples = samples_new

                    if log_gt_frames:
                        if dataset_latent:
                            # if batched chunked_task is with latent, then decode it
                            value_dict["cond_frames"] = model.decode_first_stage(
                                model.encode_first_stage(
                                    value_dict["cond_frames"].to("cuda"),
                                    sampled_modality_ids=[0],
                                ),
                                sampled_modality_ids=[0],
                            )["rgb"]  # (T, 3, H, W)

                        if isinstance(samples, dict):
                            samples_new = {}
                            for k, v in samples.items():
                                samples_new[f"{k}/image" if "rgb" in k else k] = (
                                    samples[k]
                                )
                            samples = samples_new
                        else:
                            samples = {
                                "samples-rgb/image": samples,
                            }
                        samples.update({"rgb/image": value_dict["cond_frames"]})

                    unload_model(model.first_stage_model)
                    return samples


html_header = """
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
desc = """
# Scena gradio demo

## Workflow

1. Upload an image.
2. Choose camera trajecotry and #frames, click "Render".
3. Three videos will be generated: preprocessed input, intermediate output, and final output.

> For a 80-frame video, intermediate output takes 20s, final output takes 2~3m.
> Our model currently doesn't work well with human and animal images.
"""


def main(cfg: ScenaRendererConfig, server_port: int | None = None):
    example_dir = "/weka/home-jensen/scena-image/nonsquare/"

    server = viser.ViserServer()

    @server.on_client_connect
    def _(_):
        server.scene.reset()

    renderer = ScenaRenderer(cfg, server)
    server_url = server.request_share_url()

    with gr.Blocks(head=html_header) as demo:
        task_state = gr.State()
        options_state = gr.State(
            ScenaRenderOptions(
                seed=23,
            )
        )
        with gr.Row():
            gr.Markdown(desc)
        with gr.Row():
            viewport = gr.HTML(
                f'<iframe src="{server_url}" style="display: block; margin: auto; width: 100%; height: 60vh;" frameborder="0"></iframe>',
                container=True,
                visible=False,
            )
        with gr.Row():
            with gr.Column():
                input_imgs = gr.Gallery(interactive=True, label="Input")
                shorter_handle = gr.Number(value=576, label="Resize shorter side")
                use_center_crop_handle = gr.Checkbox(
                    True,
                    label="Square crop? If not, keep aspect ratio",
                )
                preprocess_btn = gr.Button("Preprocess")
            with gr.Column():
                seed_handle = gr.Number(value=23, label="Random seed")
                seed_handle.change(
                    lambda options, seed: options.update(seed=seed),
                    inputs=[options_state, seed_handle],
                )
                # with gr.Accordion("Advanced options", open=False):
                #     text_input = gr.Textbox(label="Enter text")
                render_btn = gr.Button("Render")
            with gr.Column():
                p1_video = gr.Video(
                    label="Intermediate output", autoplay=True, loop=True
                )
                p2_video = gr.Video(label="Final output", autoplay=True, loop=True)
        preprocess_btn.click(
            renderer.preprocess,
            inputs=[input_imgs, shorter_handle, use_center_crop_handle],
            outputs=[task_state],
        )
        preprocess_btn.click(lambda: gr.HTML(visible=True), outputs=[viewport])
        render_btn.click(
            renderer.render_video,
            inputs=[task_state, options_state],
            outputs=[p1_video, p2_video],
        )

    try:
        demo.launch(
            share=True,
            allowed_paths=[example_dir, cfg.output_root],
            server_port=server_port,
        )
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    tyro.cli(main)
