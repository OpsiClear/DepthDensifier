import numpy as np
import torch
import pycolmap
from moge.model.v2 import MoGeModel
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import time
from dataclasses import dataclass, field
import tyro
import dataclasses
import logging

# Import the refiner from your source directory
from depthdensifier.depth_refiner import DepthRefiner, RefinerConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. SCRIPT CONFIGURATION
# ==============================================================================


@dataclass
class PathsConfig:
    """Configuration for input and output paths."""

    recon_path: Path = Path("data/360_v2/bicycle/sparse/0")
    image_dir: Path = Path("data/360_v2/bicycle/images")
    output_model_dir: Path = Path("results/0")


@dataclass
class MoGeConfig:
    """Configuration for the MoGe model."""

    checkpoint: Path = Path("models/moge/moge-2-vitl-normal/model.pt")


@dataclass
class ProcessingConfig:
    """Parameters for processing and densification."""

    pipeline_downsample_factor: int = 1
    """Factor to downsample images before processing. Larger is faster."""
    downsample_density: int = 32
    """Controls final point cloud density (1=densest)."""


@dataclass
class FilteringConfig:
    """Parameters for multi-view consistency filtering."""

    vote_threshold: int = 5
    """Number of votes required to remove a 'floater' point."""
    depth_threshold: float = 0.7
    """Threshold to identify a floater (projected_depth < T * refined_depth)."""
    grazing_angle_threshold: float = 0.052
    """Threshold for grazing angle filtering (cos of angle, ~87 degrees)."""


@dataclass
class ScriptConfig:
    """Main configuration for the densification script."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    moge: MoGeConfig = field(default_factory=MoGeConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)


class FloaterFilter:
    """
    A class to filter floater points from a dense point cloud using multi-view
    consistency, accelerated on the GPU with PyTorch.
    """

    def __init__(self, config: FilteringConfig, device: torch.device):
        """
        Initializes the FloaterFilter.

        :param config: A dataclass object containing filtering parameters
                       (e.g., vote_threshold, depth_threshold).
        :param device: The torch.device (e.g., 'cuda' or 'cpu') to run on.
        """
        self.config = config
        self.device = device

    def _project_points_torch(
        self,
        points3d_world: torch.Tensor,
        cam_from_world: torch.Tensor,
        K: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Projects 3D world points to 2D image coordinates using PyTorch.

        :param points3d_world: Tensor of 3D points in world coordinates (N, 3).
        :param cam_from_world: Camera-from-world transformation matrix (4, 4).
        :param K: Camera intrinsics matrix (3, 3).
        :return: A tuple of (points2d, depths) in image coordinates.
        """
        points3d_h = torch.cat(
            [points3d_world, torch.ones_like(points3d_world[:, :1])], dim=1
        )
        points_cam_h = (cam_from_world @ points3d_h.T).T

        points_cam = points_cam_h[:, :3]
        depths = points_cam[:, 2]

        # Camera to image plane projection
        points_cam_normalized = points_cam / (depths.unsqueeze(1) + 1e-8)

        points2d_h = (K @ points_cam_normalized.T).T
        points2d = points2d_h[:, :2]

        return points2d, depths

    @torch.no_grad()
    def filter(
        self,
        point_cloud: np.ndarray,
        colors: np.ndarray,
        normals: np.ndarray,
        cached_refinement_data: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filters the point cloud for floaters.

        :param point_cloud: The dense point cloud (N, 3).
        :param colors: The colors for each point (N, 3).
        :param normals: The normals for each point (N, 3).
        :param cached_refinement_data: Dictionary of refinement data per image.
        :return: A tuple of (filtered_points, filtered_colors, filtered_normals).
        """

        if not torch.cuda.is_available() or self.device.type == "cpu":
            logger.warning(
                "Filtering on CPU. This may be slow. For GPU acceleration, ensure CUDA is available."
            )

        points_t = torch.from_numpy(point_cloud).float().to(self.device)
        normals_t = torch.from_numpy(normals).float().to(self.device)
        floater_votes_t = torch.zeros(
            len(points_t), dtype=torch.int32, device=self.device
        )

        for data in tqdm(cached_refinement_data.values(), desc="Filtering Points (GPU)"):
            refined_depth = (
                torch.from_numpy(data["refined_depth"]).float().to(self.device)
            )
            h, w = refined_depth.shape

            cam_from_world_t = (
                torch.from_numpy(data["image"].cam_from_world().matrix())
                .float()
                .to(self.device)
            )
            K_t = (
                torch.from_numpy(data["camera"].calibration_matrix())
                .float()
                .to(self.device)
            )

            points2d_t, depths_t = self._project_points_torch(
                points_t, cam_from_world_t, K_t
            )

            cam_center = (
                torch.from_numpy(data["image"].projection_center())
                .float()
                .to(self.device)
            )
            viewing_dirs = torch.nn.functional.normalize(
                points_t - cam_center, p=2, dim=1
            )
            dot_products = torch.sum(normals_t * -viewing_dirs, dim=1)
            not_grazing_mask = (
                dot_products > self.config.grazing_angle_threshold
            )

            u, v = points2d_t[:, 0], points2d_t[:, 1]

            mask_in_bounds = (
                (u >= 0)
                & (u < w)
                & (v >= 0)
                & (v < h)
                & (depths_t > 0)
                & not_grazing_mask
            )

            if not torch.any(mask_in_bounds):
                continue

            u_norm = (u[mask_in_bounds] / (w - 1)) * 2 - 1
            v_norm = (v[mask_in_bounds] / (h - 1)) * 2 - 1
            grid = torch.stack([u_norm, v_norm], dim=1).unsqueeze(0).unsqueeze(0)

            refined_depths_at_projections = torch.nn.functional.grid_sample(
                refined_depth.unsqueeze(0).unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze()

            projected_depths_valid = depths_t[mask_in_bounds]
            valid_lookup_mask = refined_depths_at_projections > 0

            inconsistent_mask = (
                projected_depths_valid[valid_lookup_mask]
                < self.config.depth_threshold
                * refined_depths_at_projections[valid_lookup_mask]
            )

            original_indices = torch.where(mask_in_bounds)[0]
            indices_with_valid_lookup = original_indices[valid_lookup_mask]
            inconsistent_indices = indices_with_valid_lookup[inconsistent_mask]

            floater_votes_t.scatter_add_(
                0,
                inconsistent_indices,
                torch.ones_like(inconsistent_indices, dtype=torch.int32),
            )

        points_to_keep_mask = floater_votes_t < self.config.vote_threshold
        num_removed = torch.sum(~points_to_keep_mask).item()
        num_total = len(points_t)
        if num_total > 0:
            logger.info(
                f"-> Filtering removed {num_removed} points ({num_removed / num_total * 100:.2f}%)"
            )

        keep_mask_np = points_to_keep_mask.cpu().numpy()
        return (
            point_cloud[keep_mask_np],
            colors[keep_mask_np],
            normals[keep_mask_np],
        )


def unproject_points(points2D, depth, camera: pycolmap.Camera):
    """Unprojects 2D image points to 3D camera coordinates."""
    fx, fy, cx, cy = camera.params
    u, v = points2D[:, 0], points2D[:, 1]
    x_normalized = (u - cx) / fx
    y_normalized = (v - cy) / fy
    points3D_camera = np.stack(
        [x_normalized * depth, y_normalized * depth, depth], axis=-1
    )
    return points3D_camera


# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
def main(config: ScriptConfig):
    total_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. MoGe inference will be slow.")

    # --- 1. Load MoGe Model ---
    step_start_time = time.time()
    logger.info(f"Loading MoGe model ({config.moge.checkpoint})...")
    model = MoGeModel.from_pretrained(str(config.moge.checkpoint)).to(device)
    model.eval()
    logger.info(f"-> MoGe model loaded in {time.time() - step_start_time:.2f}s.")

    # --- 2. Load COLMAP Reconstruction ---
    step_start_time = time.time()
    logger.info(f"Loading COLMAP reconstruction from {config.paths.recon_path}...")
    rec = pycolmap.Reconstruction(config.paths.recon_path)
    logger.info(
        f"Loaded model with {rec.num_reg_images()} images and {rec.num_points3D()} sparse points."
    )
    logger.info(
        f"-> COLMAP reconstruction loaded in {time.time() - step_start_time:.2f}s."
    )

    # --- 3. Initialize the Depth Refiner ---
    step_start_time = time.time()
    logger.info("Initializing Depth Refiner...")
    refiner_config = dataclasses.asdict(config.refiner)
    refiner = DepthRefiner(**refiner_config)
    logger.info(
        f"-> Depth Refiner initialized in {time.time() - step_start_time:.2f}s."
    )

    # --- 3.5. Initialize the Floater Filter ---
    step_start_time = time.time()
    logger.info("Initializing Floater Filter...")
    floater_filter = FloaterFilter(config.filtering, device)
    logger.info(
        f"-> Floater Filter initialized in {time.time() - step_start_time:.2f}s."
    )

    # --- 4. Process each image: Infer, Refine, and Densify ---
    processing_start_time = time.time()
    all_dense_points = []
    all_dense_colors = []
    all_dense_normals = []
    all_unrefined_points = []
    cached_refinement_data = {}
    num_points = 0
    image_list = [img for img in rec.images.values() if img.has_pose]
    for image in tqdm(image_list, desc="Refining and Densifying"):
        per_image_start_time = time.time()

        step_start_time = time.time()
        point3D_ids = [p.point3D_id for p in image.points2D if p.has_point3D()]
        if len(point3D_ids) == 0:
            continue

        points3D_world = np.array([rec.points3D[pid].xyz for pid in point3D_ids])
        if refiner_config["verbose"] > 0:
            logger.debug(f"  - Get sparse points: {time.time() - step_start_time:.2f}s")

        # --- Load and Downsample Image FIRST ---
        step_start_time = time.time()
        image_path = config.paths.image_dir / image.name
        pil_image = Image.open(image_path).convert("RGB")

        w_orig, h_orig = pil_image.size
        new_w = w_orig // config.processing.pipeline_downsample_factor
        new_h = h_orig // config.processing.pipeline_downsample_factor

        pil_image_rescaled = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        img_tensor = (
            torch.from_numpy(np.array(pil_image_rescaled))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        img_tensor = img_tensor.to(device)
        if refiner_config["verbose"] > 0:
            logger.debug(
                f"  - Image Load & Downsample: {time.time() - step_start_time:.2f}s"
            )

        # --- Run MoGe Inference on the smaller image ---
        step_start_time = time.time()
        with torch.no_grad():
            moge_output = model.infer(img_tensor)
        if refiner_config["verbose"] > 0:
            logger.debug(f"  - MoGe Inference: {time.time() - step_start_time:.2f}s")

        moge_depth = moge_output["depth"].squeeze(0).cpu().numpy()
        moge_normal = moge_output["normal"].squeeze(0).cpu().numpy()
        moge_mask = moge_output["mask"].squeeze(0).cpu().numpy()

        # --- Refine the depth map at the processing resolution ---
        step_start_time = time.time()
        camera = rec.cameras[image.camera_id]
        camera.rescale(new_width=new_w, new_height=new_h)
        if refiner_config["verbose"] > 0:
            logger.debug(
                f"\n--- Refining depth for {image.name} (ID: {image.image_id}) ---"
            )

        cam_from_world_mat = image.cam_from_world().matrix()[:3, :]
        K_mat = camera.calibration_matrix()
        refinement_results = refiner.refine_depth(
            depth_map=moge_depth,
            normal_map=moge_normal,
            points3D=points3D_world,
            cam_from_world=cam_from_world_mat,
            K=K_mat,
            mask=moge_mask,
        )
        if refiner_config["verbose"] > 0:
            logger.debug(f"  - Depth Refinement: {time.time() - step_start_time:.2f}s")

        refined_depth = refinement_results["refined_depth"]

        # Apply the MoGe mask to the refined depth map before caching and densification.
        # This ensures we only work with depths from the object of interest going forward.
        refined_depth[~moge_mask] = 0

        # Cache data for filtering step
        cached_refinement_data[image.image_id] = {
            "refined_depth": refined_depth,
            "camera": camera,
            "image": image,
        }

        # --- Densify using the refined (and already downsampled) depth map ---
        step_start_time = time.time()
        h, w = refined_depth.shape
        pixels_y, pixels_x = np.mgrid[
            0 : h : config.processing.downsample_density,
            0 : w : config.processing.downsample_density,
        ]

        # The refined depth map already has the MoGe mask applied.
        # We create points only where the depth is non-zero.
        valid_pixels = refined_depth[pixels_y, pixels_x] > 0

        pixels_x_valid, pixels_y_valid = pixels_x[valid_pixels], pixels_y[valid_pixels]

        # Get colors for the valid pixels from the rescaled image
        img_np_rescaled = np.array(pil_image_rescaled)
        colors = img_np_rescaled[pixels_y_valid, pixels_x_valid]

        # Get normals for the valid pixels.
        # The normal map has a shape of (H, W, 3).
        normals = moge_normal[pixels_y_valid, pixels_x_valid]

        # --- [DEBUG] Unproject unrefined depth for comparison ---
        depth_values_unrefined = moge_depth[pixels_y_valid, pixels_x_valid]
        points2D_unrefined = np.stack([pixels_x_valid, pixels_y_valid], axis=-1)
        points3D_camera_unrefined = unproject_points(
            points2D_unrefined, depth_values_unrefined, camera
        )
        points3D_world_unrefined = (
            image.cam_from_world().inverse() * points3D_camera_unrefined
        )
        all_unrefined_points.append(points3D_world_unrefined)

        depth_values = refined_depth[pixels_y_valid, pixels_x_valid]

        points2D = np.stack([pixels_x_valid, pixels_y_valid], axis=-1)
        points3D_camera = unproject_points(points2D, depth_values, camera)
        points3D_world = image.cam_from_world().inverse() * points3D_camera
        if refiner_config["verbose"] > 0:
            logger.debug(
                f"  - Densification calculation: {time.time() - step_start_time:.2f}s"
            )

        append_start_time = time.time()
        all_dense_points.append(points3D_world)
        all_dense_colors.append(colors)
        all_dense_normals.append(normals)
        if refiner_config["verbose"] > 0:
            logger.debug(
                f"  - Appending points to list: {time.time() - append_start_time:.2f}s"
            )

        num_points += len(points3D_world)
        logger.info(f"number of dense points: {num_points}")

        # --- Explicitly clean up GPU memory to prevent slowdown ---
        cleanup_start_time = time.time()
        del img_tensor, moge_output, refinement_results
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if refiner_config["verbose"] > 0:
            logger.debug(
                f"  - GPU Memory Cleanup: {time.time() - cleanup_start_time:.2f}s"
            )

        if refiner_config["verbose"] > 0:
            logger.debug(
                f"-> Processed image ({image.name}) in {time.time() - per_image_start_time:.2f}s"
            )

    logger.info(
        f"-> Image processing loop finished in {time.time() - processing_start_time:.2f}s."
    )
    rec = pycolmap.Reconstruction(config.paths.recon_path)

    # --- 5. Save the final results ---
    saving_start_time = time.time()
    if all_dense_points:
        step_start_time = time.time()
        final_point_cloud = np.concatenate(all_dense_points, axis=0)
        final_colors = np.concatenate(all_dense_colors, axis=0)
        final_normals = np.concatenate(all_dense_normals, axis=0)
        logger.info(
            f"-> Point cloud concatenated in {time.time() - step_start_time:.2f}s."
        )

        # --- Filter point cloud based on multi-view consistency (GPU) ---
        logger.info("--- Filtering point cloud for geometric consistency (GPU) ---")
        filter_start_time = time.time()

        final_point_cloud, final_colors, final_normals = floater_filter.filter(
            point_cloud=final_point_cloud,
            colors=final_colors,
            normals=final_normals,
            cached_refinement_data=cached_refinement_data,
        )

        logger.info(f"-> Filtering finished in {time.time() - filter_start_time:.2f}s.")

        # --- Update the reconstruction object and save as COLMAP model ---
        step_start_time = time.time()

        step_start_time = time.time()
        logger.info(f"Adding {len(final_point_cloud)} new dense points...")
        for i in range(len(final_point_cloud)):
            xyz = final_point_cloud[i]
            rgb = final_colors[i]
            rec.add_point3D(xyz=xyz, track=pycolmap.Track(), color=rgb)
        logger.info(f"-> New points added in {time.time() - step_start_time:.2f}s.")

        step_start_time = time.time()
        config.paths.output_model_dir.mkdir(parents=True, exist_ok=True)
        rec.write_binary(str(config.paths.output_model_dir))
        logger.info(f"COLMAP binary model saved to: {config.paths.output_model_dir}")
        logger.info(f"-> COLMAP model written in {time.time() - step_start_time:.2f}s.")
    else:
        logger.warning("No dense points were generated. Skipping save.")

    logger.info(f"-> Saving finished in {time.time() - saving_start_time:.2f}s.")
    logger.info(f"Total script execution time: {time.time() - total_start_time:.2f}s")


if __name__ == "__main__":
    tyro.cli(main)
