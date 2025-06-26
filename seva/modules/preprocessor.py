import contextlib
import os
import os.path as osp
import sys
from typing import cast

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F


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
    def __init__(
        self, device: str | torch.device = "cuda", conf_threshold: float = 4.0
    ):
        submodule_path = osp.realpath(
            osp.join(osp.dirname(__file__), "../../third_party/vggt/")
        )
        if submodule_path not in sys.path:
            sys.path.insert(0, submodule_path)
        try:
            from vggt.models.vggt import VGGT  # type: ignore[import]
            from vggt.utils.load_fn import load_and_preprocess_images_square  # type: ignore[import]
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore[import]
            from vggt.utils.geometry import unproject_depth_map_to_point_map  # type: ignore[import]
            from vggt.utils.helper import create_pixel_coordinate_grid  # type: ignore[import]
            # from vggt.dependency.track_predict import predict_tracks # type: ignore[import]

        except ImportError:
            raise ImportError(
                "Missing required submodule: 'vggt'. Please ensure that all submodules are properly set up.\n\n"
                "To initialize them, run the following command in the project root:\n"
                "  git submodule update --init --recursive"
            )

        self.device = torch.device(device)
        self.dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        self.model = VGGT()
        self.state_dict = torch.load(
            "/home/atopaloglu21/stable-virtual-camera/third_party/vggt/weights/model_tracker_fixed_e20.pt",
            map_location="cpu",
        )
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
        images = F.interpolate(
            images, size=(resolution, resolution), mode="bilinear", align_corners=False
        )

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = model.aggregator(images)

            # Predict Cameras
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = self._pose_encoding_to_extri_intri(
                pose_enc, images.shape[-2:]
            )
            # Predict Depth Maps
            depth_map, depth_conf = self.model.depth_head(
                aggregated_tokens_list, images, ps_idx
            )

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
        list[np.ndarray],
        np.ndarray,
        np.ndarray,
        list[np.ndarray],
        list[np.ndarray],
        tuple[int, int],
        float,
    ]:
        if not img_paths:
            raise ValueError("Input image paths list cannot be empty.")

        num_images = len(img_paths)

        ori_imgs_list = []
        ori_img_whs_list = []  # List of [W, H] floats
        for p in img_paths:
            img = iio.imread(p)  # np.ndarray, HxWxC or HxW.
            ori_imgs_list.append(img)
            # Ensure shape is at least 2D for shape access
            if img.ndim < 2:
                raise ValueError(f"Image {p} has insufficient dimensions: {img.ndim}")
            ori_img_whs_list.append([float(img.shape[1]), float(img.shape[0])])  # W, H

        first_img_w, first_img_h = ori_img_whs_list[0]
        self.img_load_resolution = float(max(first_img_w, first_img_h))

        ori_imgs = [iio.imread(p) for p in img_paths]
        imgs = cast(list, ori_imgs)

        images_loaded_sq_tensor, original_coords_tensor = (
            self._load_and_preprocess_images_square(
                img_paths, target_size=self.img_load_resolution
            )
        )
        images_loaded_sq_tensor = images_loaded_sq_tensor.to(self.device)
        original_coords_tensor = original_coords_tensor.to(self.device)
        print(f"Loaded {len(images_loaded_sq_tensor)}")

        extrinsic_w2c_np, intrinsic_518_np, depth_map_518_np, depth_conf_518_np = (
            self.run_VGGT(
                self.model,
                images_loaded_sq_tensor,
                self.dtype,
                self.vggt_fixed_resolution,
            )
        )
        points_3d = self._unproject_depth_map_to_point_map(
            depth_map_518_np, extrinsic_w2c_np, intrinsic_518_np
        )

        points_rgb = F.interpolate(
            images_loaded_sq_tensor,
            size=(self.vggt_fixed_resolution, self.vggt_fixed_resolution),
            mode="bilinear",
            align_corners=False,
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
        intrinsic_518_np[:, 1, :] *= s_load_factor_h

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


import numpy as np
from collections import namedtuple
import subprocess

Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
ImageCOLMAP = namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)


class ColmapPipeline(object):
    def __init__(self):
        try:
            # Check if the 'colmap' command is available.
            subprocess.run(
                ["colmap", "help"], check=True, capture_output=True, text=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError(
                "\n--- ERROR ---\n"
                "'colmap' command not found. Please ensure COLMAP is installed "
                "and its executable is in your system's PATH."
            )

    def _qvec2rotmat(self, qvec: np.ndarray) -> np.ndarray:
        """
        Convert a quaternion vector to a 3x3 rotation matrix.
        """
        return np.array(
            [
                [
                    1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                    2 * qvec[1] * qvec[2] - 2 * qvec[3] * qvec[0],
                    2 * qvec[1] * qvec[3] + 2 * qvec[2] * qvec[0],
                ],
                [
                    2 * qvec[1] * qvec[2] + 2 * qvec[3] * qvec[0],
                    1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                    2 * qvec[2] * qvec[3] - 2 * qvec[1] * qvec[0],
                ],
                [
                    2 * qvec[1] * qvec[3] - 2 * qvec[2] * qvec[0],
                    2 * qvec[2] * qvec[3] + 2 * qvec[1] * qvec[0],
                    1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
                ],
            ]
        )

    def _read_cameras_text(self, path: str) -> dict:
        """
        Parses the cameras.txt file from a COLMAP text export.
        """
        cameras = {}
        with open(path, "r") as fid:
            for line in fid:
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    parts = line.split()
                    camera_id = int(parts[0])
                    model = parts[1]
                    width = int(parts[2])
                    height = int(parts[3])
                    params = np.array([float(p) for p in parts[4:]])
                    cameras[camera_id] = Camera(
                        id=camera_id,
                        model=model,
                        width=width,
                        height=height,
                        params=params,
                    )
        return cameras

    def _read_images_text(self, path: str) -> dict:
        """
        Parses the images.txt file from a COLMAP text export.
        """
        images = {}
        with open(path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    parts = line.split()
                    image_id = int(parts[0])
                    qvec = np.array(tuple(map(float, parts[1:5])))
                    tvec = np.array(tuple(map(float, parts[5:8])))
                    camera_id = int(parts[8])
                    image_name = parts[9]

                    line = fid.readline().strip()
                    points2D = np.array([float(p) for p in line.split()])

                    # COLMAP format is X, Y, POINT3D_ID
                    xys = points2D[0::3]
                    ys = points2D[1::3]
                    point3D_ids = np.array(points2D[2::3], dtype=np.int64)

                    images[image_id] = ImageCOLMAP(
                        id=image_id,
                        qvec=qvec,
                        tvec=tvec,
                        camera_id=camera_id,
                        name=image_name,
                        xys=np.column_stack((xys, ys)),
                        point3D_ids=point3D_ids,
                    )
        return images

    def _read_points3d_text(self, path: str) -> dict:
        """
        Parses the points3D.txt file from a COLMAP text export.
        """
        points3D = {}
        with open(path, "r") as fid:
            for line in fid:
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    parts = line.split()
                    point3D_id = int(parts[0])
                    xyz = np.array(tuple(map(float, parts[1:4])))
                    rgb = np.array(tuple(map(int, parts[4:7])))
                    # We don't need error or track info for this pipeline
                    points3D[point3D_id] = {"xyz": xyz, "rgb": rgb}
        return points3D

    def infer_cameras_and_points(
        self,
        colmap_project_dir: str,
        every_nth: int = 20,
    ) -> tuple[
        list[np.ndarray], np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]
    ]:
        """
        Loads data from a COLMAP project directory.

        This method automatically converts COLMAP .bin files to .txt, then extracts
        images, intrinsics (Ks), camera-to-world extrinsics (c2ws), and the
        corresponding 3D point cloud for every nth image.

        Args:
            colmap_project_dir (str): The path to the COLMAP project directory.
                                      This directory should contain 'sparse/0' and 'images'.
            every_nth (int): The interval for sampling images from the reconstruction.

        Returns:
            A tuple containing:
            - imgs (list[np.ndarray]): List of loaded image arrays (H, W, C).
            - Ks (np.ndarray): (N, 3, 3) array of intrinsic matrices.
            - c2ws (np.ndarray): (N, 3, 4) array of camera-to-world transformation matrices.
            - points (list[np.ndarray]): List of (M_i, 3) arrays of 3D point coordinates for each image.
            - point_colors (list[np.ndarray]): List of (M_i, 3) arrays of RGB colors for each point.
        """
        # Define paths
        bin_dir = os.path.join(colmap_project_dir, "sparse", "0")
        txt_dir = os.path.join(colmap_project_dir, "sparse", "0_txt")
        images_dir = os.path.join(colmap_project_dir, "images")

        # --- Automatic BIN to TXT Conversion ---
        if not os.path.exists(os.path.join(txt_dir, "cameras.txt")):
            print(
                f"Text files not found in {txt_dir}. Checking for binary files in {bin_dir}."
            )
            if not os.path.exists(os.path.join(bin_dir, "cameras.bin")):
                raise FileNotFoundError(
                    f"COLMAP data not found in {bin_dir} or {txt_dir}."
                )

            print("Binary files found. Running COLMAP model_converter...")
            os.makedirs(txt_dir, exist_ok=True)

            try:
                command = [
                    "colmap",
                    "model_converter",
                    "--input_path",
                    bin_dir,
                    "--output_path",
                    txt_dir,
                    "--output_type",
                    "TXT",
                ]
                result = subprocess.run(
                    command, check=True, capture_output=True, text=True
                )
                print("COLMAP model_converter completed successfully.")
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print("\n--- ERROR ---")
                print("COLMAP model_converter failed to run.")
                print(f"Return Code: {e.returncode}")
                print(f"Output:\n{e.stdout}")
                print(f"Error:\n{e.stderr}")
                raise
        else:
            print(f"Using existing text files from {txt_dir}.")

        # --- Data Loading from TXT files ---
        cameras_path = os.path.join(txt_dir, "cameras.txt")
        images_path = os.path.join(txt_dir, "images.txt")
        points3d_path = os.path.join(txt_dir, "points3D.txt")

        print("Parsing COLMAP text files...")
        cameras = self._read_cameras_text(cameras_path)
        images_data = self._read_images_text(images_path)
        points3D_data = self._read_points3d_text(points3d_path)

        images_list = []
        intrinsics_list = []
        c2w_list = []
        points_list = []
        point_colors_list = []

        sorted_images = sorted(images_data.values(), key=lambda img: img.name)
        sampled_images = sorted_images[::every_nth]
        print(
            f"Found {len(sorted_images)} total images, processing {len(sampled_images)} images (every {every_nth})."
        )

        for image in sampled_images:
            # Load image
            img_path = os.path.join(images_dir, image.name)
            if not os.path.exists(img_path):
                print(f"Warning: Image file not found at {img_path}, skipping.")
                continue
            images_list.append(iio.imread(img_path))

            camera = cameras[image.camera_id]

            # --- Intrinsics (K) ---
            K = np.eye(3, dtype=np.float32)
            if camera.model == "SIMPLE_PINHOLE":
                K[0, 0] = K[1, 1] = camera.params[0]  # f
                K[0, 2], K[1, 2] = camera.params[1], camera.params[2]  # cx, cy
            elif camera.model == "PINHOLE":
                K[0, 0], K[1, 1] = camera.params[0], camera.params[1]  # fx, fy
                K[0, 2], K[1, 2] = camera.params[2], camera.params[3]  # cx, cy
            else:
                # Fallback for other models, might not be perfect
                print(
                    f"Warning: Camera model '{camera.model}' not fully supported. Using best-effort pinhole params."
                )
                K[0, 0], K[1, 1] = camera.params[0], camera.params[1]
                K[0, 2], K[1, 2] = camera.params[2], camera.params[3]
            intrinsics_list.append(K)

            # --- Extrinsics (c2w) ---
            R = self._qvec2rotmat(image.qvec)
            t = image.tvec.reshape(3, 1)

            # c2w is [R.T | -R.T @ t]
            R_inv = R.T
            t_inv = -R_inv @ t
            c2w = np.hstack((R_inv, t_inv)).astype(np.float32)

            c2w_h = np.vstack((c2w, np.array([[0, 0, 0, 1]], dtype=np.float32)))

            c2w_list.append(c2w_h)

            current_points = []
            current_colors = []
            for point3D_id in image.point3D_ids:
                if point3D_id != -1 and point3D_id in points3D_data:
                    point_data = points3D_data[point3D_id]
                    current_points.append(point_data["xyz"])
                    current_colors.append(point_data["rgb"])

            points_list.append(np.array(current_points, dtype=np.float32))
            point_colors_list.append(np.array(current_colors, dtype=np.uint8))

        Ks = np.array(intrinsics_list, dtype=np.float32)
        c2ws = np.array(c2w_list, dtype=np.float32)

        return images_list, Ks, c2ws, points_list, point_colors_list
