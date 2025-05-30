import contextlib
import os
import os.path as osp
import sys
from typing import cast

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

from seva.geometry import to_hom_pose  
from PIL import Image

class Dust3rPipeline(object):
    def __init__(self, device: str | torch.device = "cuda"):
        submodule_path = osp.realpath(
            osp.join(osp.dirname(__file__), "../../third_party/dust3r/")
        )
        if submodule_path not in sys.path:
            sys.path.insert(0, submodule_path)
        try:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                from dust3r.cloud_opt import (  # type: ignore[import]
                    GlobalAlignerMode,
                    global_aligner,
                )
                from dust3r.image_pairs import make_pairs  # type: ignore[import]
                from dust3r.inference import inference  # type: ignore[import]
                from dust3r.model import AsymmetricCroCo3DStereo  # type: ignore[import]
                from dust3r.utils.image import load_images  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "Missing required submodule: 'dust3r'. Please ensure that all submodules are properly set up.\n\n"
                "To initialize them, run the following command in the project root:\n"
                "  git submodule update --init --recursive"
            )

        self.device = torch.device(device)
        self.model = AsymmetricCroCo3DStereo.from_pretrained(
            "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        ).to(self.device)

        self._GlobalAlignerMode = GlobalAlignerMode
        self._global_aligner = global_aligner
        self._make_pairs = make_pairs
        self._inference = inference
        self._load_images = load_images

    def infer_cameras_and_points(
        self,
        img_paths: list[str],
        Ks: list[list] = None,
        c2ws: list[list] = None,
        batch_size: int = 16,
        schedule: str = "cosine",
        lr: float = 0.01,
        niter: int = 500,
        min_conf_thr: int = 3,
    ) -> tuple[
        list[np.ndarray], np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]
    ]:
        num_img = len(img_paths)
        if num_img == 1:
            print("Only one image found, duplicating it to create a stereo pair.")
            img_paths = img_paths * 2

        images = self._load_images(img_paths, size=512)
        pairs = self._make_pairs(
            images,
            scene_graph="complete",
            prefilter=None,
            symmetrize=True,
        )
        output = self._inference(pairs, self.model, self.device, batch_size=batch_size)

        ori_imgs = [iio.imread(p) for p in img_paths]
        ori_img_whs = np.array([img.shape[1::-1] for img in ori_imgs])
        img_whs = np.concatenate([image["true_shape"][:, ::-1] for image in images], 0)

        scene = self._global_aligner(
            output,
            device=self.device,
            mode=self._GlobalAlignerMode.PointCloudOptimizer,
            same_focals=True,
            optimize_pp=False,  # True,
            min_conf_thr=min_conf_thr,
        )

        # if Ks is not None:
        #     scene.preset_focal(
        #         torch.tensor([[K[0, 0], K[1, 1]] for K in Ks])
        #     )

        if c2ws is not None:
            scene.preset_pose(c2ws)

        _ = scene.compute_global_alignment(
            init="msp", niter=niter, schedule=schedule, lr=lr
        )

        imgs = cast(list, scene.imgs)
        Ks = scene.get_intrinsics().detach().cpu().numpy().copy()
        c2ws = scene.get_im_poses().detach().cpu().numpy()  # type: ignore
        pts3d = [x.detach().cpu().numpy() for x in scene.get_pts3d()]  # type: ignore
        if num_img > 1:
            masks = [x.detach().cpu().numpy() for x in scene.get_masks()]
            points = [p[m] for p, m in zip(pts3d, masks)]
            point_colors = [img[m] for img, m in zip(imgs, masks)]
        else:
            points = [p.reshape(-1, 3) for p in pts3d]
            point_colors = [img.reshape(-1, 3) for img in imgs]

        # Convert back to the original image size.
        imgs = ori_imgs
        Ks[:, :2, -1] *= ori_img_whs / img_whs
        Ks[:, :2, :2] *= (ori_img_whs / img_whs).mean(axis=1, keepdims=True)[..., None]

        return imgs, Ks, c2ws, points, point_colors

class VggtPipeline(object):
    def __init__(self, device: str | torch.device = "cuda", conf_threshold: float = 4.0):
        submodule_path = osp.realpath(
            osp.join(osp.dirname(__file__), "../../third_party/vggt/")
        )
        if submodule_path not in sys.path:
            sys.path.insert(0, submodule_path)
        try:
            from vggt.models.vggt import VGGT # type: ignore[import]
            from vggt.utils.load_fn import load_and_preprocess_images_square # type: ignore[import]
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri # type: ignore[import]
            from vggt.utils.geometry import unproject_depth_map_to_point_map # type: ignore[import]
            from vggt.utils.helper import create_pixel_coordinate_grid # type: ignore[import]
            # from vggt.dependency.track_predict import predict_tracks # type: ignore[import]

        except ImportError:
            raise ImportError(
                "Missing required submodule: 'vggt'. Please ensure that all submodules are properly set up.\n\n"
                "To initialize them, run the following command in the project root:\n"
                "  git submodule update --init --recursive"
            )
        
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        self.model = VGGT()
        self.state_dict = torch.load("/home/atopaloglu21/stable-virtual-camera/third_party/vggt/weights/model_tracker_fixed_e20.pt", map_location="cpu")
        self.model.load_state_dict(self.state_dict)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        self._unproject_depth_map_to_point_map = unproject_depth_map_to_point_map
        self._pose_encoding_to_extri_intri = pose_encoding_to_extri_intri
        self._load_and_preprocess_images_square = load_and_preprocess_images_square
        self._create_pixel_coordinate_grid = create_pixel_coordinate_grid
        self.vis_threshold = 0.2
        self.max_query_pts = 2048
        self.query_frame_num = 5
        self.fine_tracking = True
        self.keypoint_extractor = "aliked+sp"

        self.vggt_fixed_resolution = 518
        self.img_load_resolution = None
        
        self.conf_threshold = conf_threshold

    def run_VGGT(self, model, images, dtype, resolution=518):
        # images: [B, 3, H, W]
        assert len(images.shape) == 4
        assert images.shape[1] == 3

        # hard-coded to use 518 for VGGT
        images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = model.aggregator(images)

            # Predict Cameras
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = self._pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            # Predict Depth Maps
            depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)

        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().numpy()
        depth_map = depth_map.squeeze(0).cpu().numpy()
        depth_conf = depth_conf.squeeze(0).cpu().numpy()
        return extrinsic, intrinsic, depth_map, depth_conf        


    @torch.no_grad()
    def infer_cameras_and_points(
        self,
        img_paths: list[str],
        # Unused parameters, kept for API compatibility with Dust3rPipeline
        Ks: list[list] = None,
        c2ws: list[list] = None,
    ) -> tuple[
        list[np.ndarray], np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], tuple[int, int], float
    ]:
        if not img_paths:
            raise ValueError("Input image paths list cannot be empty.")
        
        num_images = len(img_paths)

        ori_imgs_list = []
        ori_img_whs_list = [] # List of [W, H] floats
        for p in img_paths:
            img = iio.imread(p) # np.ndarray, HxWxC or HxW.
            ori_imgs_list.append(img)
            # Ensure shape is at least 2D for shape access
            if img.ndim < 2:
                raise ValueError(f"Image {p} has insufficient dimensions: {img.ndim}")
            ori_img_whs_list.append([float(img.shape[1]), float(img.shape[0])]) # W, H
        
        first_img_w, first_img_h = ori_img_whs_list[0]
        self.img_load_resolution = float(max(first_img_w, first_img_h))

        ori_imgs = [iio.imread(p) for p in img_paths]
        imgs = cast(list, ori_imgs)
        
        images_loaded_sq_tensor, original_coords_tensor = self._load_and_preprocess_images_square(img_paths, target_size=self.img_load_resolution)
        images_loaded_sq_tensor = images_loaded_sq_tensor.to(self.device)
        original_coords_tensor = original_coords_tensor.to(self.device)
        print(f"Loaded {len(images_loaded_sq_tensor)}")

        extrinsic_w2c_np, intrinsic_518_np, depth_map_518_np, depth_conf_518_np = self.run_VGGT(self.model, images_loaded_sq_tensor, self.dtype, self.vggt_fixed_resolution)
        points_3d = self._unproject_depth_map_to_point_map(depth_map_518_np, extrinsic_w2c_np, intrinsic_518_np)

        num_frames, height, width, _ = points_3d.shape
        points_rgb = F.interpolate(
            images_loaded_sq_tensor, size=(self.vggt_fixed_resolution, self.vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # points_xyf = self._create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf_518_np >= self.conf_threshold

        masked_points_3d = [
            points_3d[i][conf_mask[i]]  # Shape: (N_i, 3) for each image
            for i in range(points_3d.shape[0])
        ]

        masked_points_rgb = [
            points_rgb[i][conf_mask[i]]  # Shape: (N_i, 3) for each image
            for i in range(points_3d.shape[0])
        ]

        # points_3d = points_3d[conf_mask]
        # points_xyf = points_xyf[conf_mask]
        # points_rgb = points_rgb[conf_mask]

        s_load_factor_w = first_img_w / self.vggt_fixed_resolution
        s_load_factor_h = first_img_h / self.vggt_fixed_resolution

        intrinsic_518_np[:, 0, :] *= s_load_factor_w 
        intrinsic_518_np[:, 1, 1] = intrinsic_518_np[:, 0, 0]

        Ks = intrinsic_518_np

        c2ws = np.zeros((num_images, 4, 4), dtype=np.float32)
        for i in range(num_images):
            R_w2c = extrinsic_w2c_np[i, :3, :3]
            t_w2c = extrinsic_w2c_np[i, :3, 3:]
            R_c2w = R_w2c.T
            t_c2w = -R_w2c.T @ t_w2c
            
            c2w_matrix = np.eye(4, dtype=np.float32)
            c2w_matrix[:3, :3] = R_c2w
            c2w_matrix[:3, 3] = t_c2w.flatten()
            c2ws[i] = c2w_matrix
        
        return imgs, Ks, c2ws, masked_points_3d, masked_points_rgb