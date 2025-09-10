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

# Import modules from depthdensifier
from depthdensifier.depth_refiner import DepthRefiner, RefinerConfig
from depthdensifier.floater_filter import FloaterFilter, FloaterFilterConfig
from depthdensifier.utils import unproject_points

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
class PostProcessingConfig:
    """Parameters for post-processing."""

    enable_downsampling: bool = True
    """Enable downsampling of the final point cloud if it's too large."""
    max_points: int = 500_000
    """Maximum number of points in the final cloud after downsampling."""


@dataclass
class ScriptConfig:
    """Main configuration for the densification script."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    moge: MoGeConfig = field(default_factory=MoGeConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    filtering: FloaterFilterConfig = field(default_factory=FloaterFilterConfig)
    post_processing: PostProcessingConfig = field(default_factory=PostProcessingConfig)


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

    # --- 3.5. Pre-compute camera matrices for filtering ---
    step_start_time = time.time()
    logger.info("Pre-computing camera matrices...")
    camera_matrices_cache = {}
    
    for image in rec.images.values():
        if image.has_pose:
            camera = rec.cameras[image.camera_id]
            cam_center = image.projection_center()
            cam_from_world = image.cam_from_world().matrix()
            
            camera_matrices_cache[image.image_id] = {
                "original_camera": camera,
                "cam_center": cam_center,
                "cam_from_world": cam_from_world,
                "image": image,
            }
    
    logger.info(
        f"-> Camera matrices cached for {len(camera_matrices_cache)} images "
        f"in {time.time() - step_start_time:.2f}s."
    )

    # --- 4. Process each image: Infer, Refine, and Densify ---
    processing_start_time = time.time()
    all_dense_points = []
    all_dense_colors = []
    all_dense_normals = []
    cached_refinement_data = {}
    num_points = 0
    image_list = [img for img in rec.images.values() if img.has_pose]

    logger.info(f"Processing {len(image_list)} images with poses.")

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

        # --- Filter point cloud based on multi-view consistency ---
        logger.info("--- Filtering point cloud for geometric consistency ---")
        filter_start_time = time.time()
        
        # Initialize floater filter
        floater_filter = FloaterFilter(config.filtering)
        
        # Prepare camera data
        camera_data = floater_filter.prepare_camera_data(
            cached_refinement_data, camera_matrices_cache
        )
        
        # Filter points
        points_to_keep_mask, num_removed = floater_filter.filter_points(
            final_point_cloud, final_normals, camera_data, show_progress=True
        )
        
        # Apply the filtering mask
        final_point_cloud = final_point_cloud[points_to_keep_mask]
        final_colors = final_colors[points_to_keep_mask]
        final_normals = final_normals[points_to_keep_mask]
        
        logger.info(
            f"-> Filtering removed {num_removed} points ({num_removed / len(points_to_keep_mask) * 100:.2f}%)"
        )
        logger.info(f"-> Filtering finished in {time.time() - filter_start_time:.2f}s.")

        # --- Downsample point cloud if it's too large ---
        if (
            config.post_processing.enable_downsampling
            and len(final_point_cloud) > config.post_processing.max_points
        ):
            logger.info(
                f"Point cloud has {len(final_point_cloud)} points. "
                f"Randomly downsampling to {config.post_processing.max_points}..."
            )
            downsample_start_time = time.time()

            indices = np.random.choice(
                len(final_point_cloud), config.post_processing.max_points, replace=False
            )

            final_point_cloud = final_point_cloud[indices]
            final_colors = final_colors[indices]
            final_normals = final_normals[indices]

            logger.info(
                f"Downsampled to {len(final_point_cloud)} points in "
                f"{time.time() - downsample_start_time:.2f}s."
            )

        # --- Update the reconstruction object and save as COLMAP model ---
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
