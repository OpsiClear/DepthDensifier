#!/usr/bin/env python3
"""Pipeline script using point_sampler for intelligent point sampling instead of grid-based approach."""

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
from depthdensifier.point_sampler import sample_points
from depthdensifier.utils import unproject_points

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class PathsConfig:
    """Configuration for input and output paths."""

    recon_path: Path = Path("data/360_v2/bicycle/sparse/0")
    image_dir: Path = Path("data/360_v2/bicycle/images")
    output_model_dir: Path = Path("results/point_sampler/bicycle/sparse/0")


@dataclass
class MoGeConfig:
    """Configuration for the MoGe model."""

    checkpoint: Path = Path("models/moge/moge-2-vitl-normal/model.pt")


@dataclass
class SamplingConfig:
    """Configuration for point sampling."""

    num_points_per_image: int = 15000
    """Number of points to sample per image."""
    sampling_strategy: str = "mixed"
    """Sampling strategy: 'random', 'edges', or 'mixed'."""
    edge_weight: float = 0.5
    """Weight for edge-based sampling (0.5 random, 0.5 gradient-based)."""
    use_gpu: bool = True
    """Use GPU for point sampling if available."""
    seed: int | None = None
    """Random seed for reproducible sampling."""


@dataclass
class ProcessingConfig:
    """Parameters for processing and densification."""

    pipeline_downsample_factor: int = 1
    """Factor to downsample images before processing."""
    batch_size: int = 4
    """Number of images to process in each GPU batch."""
    gpu_cache_clear_interval: int = 10
    """Clear GPU cache every N batches to prevent memory buildup."""
    skip_every_n_images: int = 1
    """Process every Nth image (1=all, 2=every other, 3=every third, etc.)."""


@dataclass
class ScriptConfig:
    """Main configuration for the densification script."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    moge: MoGeConfig = field(default_factory=MoGeConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    filtering: FloaterFilterConfig = field(default_factory=FloaterFilterConfig)


def process_batch(
    batch_data: list[tuple[pycolmap.Image, Path]],
    model: MoGeModel,
    rec: pycolmap.Reconstruction,
    refiner: DepthRefiner,
    config: ScriptConfig,
    sparse_points_cache: dict,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict]:
    """Process a batch of images through MoGe and refinement with point sampling."""

    # Load and preprocess images
    batch_tensors = []
    batch_images = []
    batch_pil_images = []
    new_w = None
    new_h = None

    for image, image_path in batch_data:
        pil_image = Image.open(image_path).convert("RGB")
        w_orig, h_orig = pil_image.size
        new_w = w_orig // config.processing.pipeline_downsample_factor
        new_h = h_orig // config.processing.pipeline_downsample_factor
        pil_image_rescaled = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        img_tensor = torch.from_numpy(np.array(pil_image_rescaled)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)

        batch_tensors.append(img_tensor)
        batch_images.append(image)
        batch_pil_images.append(pil_image_rescaled)

    if not batch_tensors:
        return [], [], [], {}

    # Stack tensors and run MoGe inference
    batch_tensor = torch.stack(batch_tensors).to(device)

    with torch.no_grad():
        batch_outputs = model.infer(batch_tensor)

    # Process each image in the batch
    batch_dense_points = []
    batch_dense_colors = []
    batch_dense_normals = []
    cached_data = {}

    for idx, image in enumerate(batch_images):
        moge_depth = batch_outputs["depth"][idx].cpu().numpy()
        moge_normal = batch_outputs["normal"][idx].cpu().numpy()
        moge_mask = batch_outputs["mask"][idx].cpu().numpy()

        # Get cached sparse points for refinement
        if image.image_id not in sparse_points_cache:
            continue

        points3D_world = sparse_points_cache[image.image_id]

        # Setup camera at processing resolution
        camera = rec.cameras[image.camera_id]
        h, w = moge_depth.shape
        camera.rescale(new_width=w, new_height=h)

        # Refine depth
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

        refined_depth = refinement_results["refined_depth"]
        refined_depth[~moge_mask] = 0

        # Cache for filtering
        cached_data[image.image_id] = {
            "refined_depth": refined_depth,
            "camera": camera,
            "image": image,
        }

        # Use point_sampler to intelligently sample points
        img_np = np.array(batch_pil_images[idx])

        # Sample points using the configured strategy
        sample_device = (
            "cuda" if config.sampling.use_gpu and torch.cuda.is_available() else "cpu"
        )
        sample_result = sample_points(
            image=img_np,
            num_points=config.sampling.num_points_per_image,
            strategy=config.sampling.sampling_strategy,
            edge_weight=config.sampling.edge_weight,
            device=sample_device,
            return_colors=True,
            seed=config.sampling.seed,
        )

        # Get pixel coordinates (integers) for sampling
        sampled_points = sample_result["pixels"]
        sampled_colors = sample_result.get("colors")

        # Filter out points where refined depth is invalid
        valid_points = []
        valid_colors = []
        valid_normals = []

        for i, (x, y) in enumerate(sampled_points):
            x_int, y_int = int(x), int(y)

            # Check bounds and depth validity
            if 0 <= x_int < w and 0 <= y_int < h and refined_depth[y_int, x_int] > 0:
                valid_points.append([x, y])
                valid_colors.append(
                    sampled_colors[i]
                    if sampled_colors is not None
                    else img_np[y_int, x_int]
                )
                valid_normals.append(moge_normal[y_int, x_int])

        if not valid_points:
            logger.warning(f"No valid points sampled for image {image.image_id}")
            continue

        valid_points = np.array(valid_points)
        valid_colors = np.array(valid_colors)
        # Convert colors from [0, 1] float to [0, 255] uint8 if needed
        if valid_colors.dtype == np.float32 or valid_colors.dtype == np.float64:
            if valid_colors.max() <= 1.0:
                valid_colors = (valid_colors * 255).astype(np.uint8)
        valid_normals = np.array(valid_normals)

        # Get depth values for valid points
        x_coords = valid_points[:, 0].astype(int)
        y_coords = valid_points[:, 1].astype(int)
        depth_values = refined_depth[y_coords, x_coords]

        # Unproject to 3D
        points3D_camera = unproject_points(valid_points, depth_values, camera)
        points3D_world = image.cam_from_world().inverse() * points3D_camera

        batch_dense_points.append(points3D_world)
        batch_dense_colors.append(valid_colors)
        batch_dense_normals.append(valid_normals)

        logger.info(
            f"Image {image.image_id}: Sampled {len(valid_points)}/{config.sampling.num_points_per_image} valid points "
            f"(strategy: {config.sampling.sampling_strategy}, edge_weight: {config.sampling.edge_weight})"
        )

    return batch_dense_points, batch_dense_colors, batch_dense_normals, cached_data


def main(config: ScriptConfig):
    total_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. MoGe inference will be slow.")

    # Log sampling configuration
    logger.info("=== Point Sampling Configuration ===")
    logger.info(f"Strategy: {config.sampling.sampling_strategy}")
    logger.info(f"Points per image: {config.sampling.num_points_per_image}")
    logger.info(
        f"Edge weight: {config.sampling.edge_weight} (Random: {1 - config.sampling.edge_weight:.1%}, Gradient: {config.sampling.edge_weight:.1%})"
    )
    logger.info(
        f"GPU sampling: {config.sampling.use_gpu and torch.cuda.is_available()}"
    )

    # --- 1. Load MoGe Model ---
    logger.info(f"Loading MoGe model ({config.moge.checkpoint})...")
    model = MoGeModel.from_pretrained(str(config.moge.checkpoint)).to(device)
    model.eval()
    logger.info(f"-> MoGe model loaded in {time.time() - total_start_time:.2f}s.")

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

    # --- 3. Initialize Depth Refiner ---
    step_start_time = time.time()
    logger.info("Initializing Depth Refiner...")
    refiner_config = dataclasses.asdict(config.refiner)
    refiner = DepthRefiner(**refiner_config)
    logger.info(
        f"-> Depth Refiner initialized in {time.time() - step_start_time:.2f}s."
    )

    # --- 4. Pre-compute sparse points and camera matrices ---
    step_start_time = time.time()
    logger.info("Pre-computing sparse points cache...")
    sparse_points_cache = {}
    camera_matrices_cache = {}

    for image in rec.images.values():
        if not image.has_pose:
            continue

        # Cache sparse points
        point3D_ids = [p.point3D_id for p in image.points2D if p.has_point3D()]
        if point3D_ids:
            points3D_world = np.array([rec.points3D[pid].xyz for pid in point3D_ids])
            sparse_points_cache[image.image_id] = points3D_world

        # Cache camera matrices
        camera = rec.cameras[image.camera_id]
        camera_matrices_cache[image.image_id] = {
            "original_camera": camera,
            "cam_center": image.projection_center(),
            "cam_from_world": image.cam_from_world().matrix(),
            "image": image,
        }

    logger.info(
        f"-> Cached sparse points for {len(sparse_points_cache)} images "
        f"in {time.time() - step_start_time:.2f}s."
    )

    # --- 5. Process images in batches ---
    processing_start_time = time.time()
    all_dense_points = []
    all_dense_colors = []
    all_dense_normals = []
    cached_refinement_data = {}

    # Prepare image batches
    all_images = [
        (img, config.paths.image_dir / img.name)
        for img in rec.images.values()
        if img.has_pose and img.image_id in sparse_points_cache
    ]

    # Apply skip_every_n_images filter
    if config.processing.skip_every_n_images > 1:
        image_list = [
            img
            for i, img in enumerate(all_images)
            if i % config.processing.skip_every_n_images == 0
        ]
        logger.info(
            f"Skipping images: processing every {config.processing.skip_every_n_images} image(s)"
        )
        logger.info(f"Reduced from {len(all_images)} to {len(image_list)} images")
    else:
        image_list = all_images

    num_batches = (
        len(image_list) + config.processing.batch_size - 1
    ) // config.processing.batch_size

    logger.info(f"Processing {len(image_list)} images in {num_batches} batches...")
    logger.info("Using point sampling instead of grid-based approach")

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * config.processing.batch_size
        end_idx = min(start_idx + config.processing.batch_size, len(image_list))
        batch_data = image_list[start_idx:end_idx]

        # Process batch with point sampling
        batch_dense_points, batch_dense_colors, batch_dense_normals, batch_cached = (
            process_batch(
                batch_data, model, rec, refiner, config, sparse_points_cache, device
            )
        )

        # Accumulate results
        all_dense_points.extend(batch_dense_points)
        all_dense_colors.extend(batch_dense_colors)
        all_dense_normals.extend(batch_dense_normals)
        cached_refinement_data.update(batch_cached)

        # Clear GPU cache periodically
        if (batch_idx + 1) % config.processing.gpu_cache_clear_interval == 0:
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Log progress
        total_points = sum(len(p) for p in all_dense_points)
        logger.info(
            f"Processed batch {batch_idx + 1}/{num_batches}, total points: {total_points}"
        )

    logger.info(
        f"-> Image processing finished in {time.time() - processing_start_time:.2f}s."
    )

    # --- 6. Concatenate and filter points ---
    if all_dense_points:
        step_start_time = time.time()
        final_point_cloud = np.concatenate(all_dense_points, axis=0)
        final_colors = np.concatenate(all_dense_colors, axis=0)
        final_normals = np.concatenate(all_dense_normals, axis=0)

        # Ensure colors are in uint8 format for COLMAP
        if final_colors.dtype != np.uint8:
            if final_colors.max() <= 1.0:
                final_colors = (final_colors * 255).astype(np.uint8)
            else:
                final_colors = final_colors.astype(np.uint8)

        logger.info(
            f"-> Point cloud concatenated in {time.time() - step_start_time:.2f}s."
        )
        logger.info(f"Total points before filtering: {len(final_point_cloud)}")

        # Filter point cloud
        logger.info("--- Filtering point cloud for geometric consistency ---")
        filter_start_time = time.time()

        floater_filter = FloaterFilter(config.filtering)
        camera_data = floater_filter.prepare_camera_data(
            cached_refinement_data, camera_matrices_cache
        )

        points_to_keep_mask, num_removed = floater_filter.filter_points(
            final_point_cloud, final_normals, camera_data, show_progress=True
        )

        final_point_cloud = final_point_cloud[points_to_keep_mask]
        final_colors = final_colors[points_to_keep_mask]

        logger.info(
            f"-> Filtering removed {num_removed} points ({num_removed / len(points_to_keep_mask) * 100:.2f}%)"
        )
        logger.info(f"-> Filtering finished in {time.time() - filter_start_time:.2f}s.")
        logger.info(f"Final point count: {len(final_point_cloud)}")

        # --- 7. Save results ---
        step_start_time = time.time()
        logger.info(f"Adding {len(final_point_cloud)} new dense points...")

        for i in range(len(final_point_cloud)):
            xyz = final_point_cloud[i]
            rgb = final_colors[i]
            rec.add_point3D(xyz=xyz, track=pycolmap.Track(), color=rgb)

        logger.info(f"-> New points added in {time.time() - step_start_time:.2f}s.")

        # Save the model
        step_start_time = time.time()
        config.paths.output_model_dir.mkdir(parents=True, exist_ok=True)
        rec.write_binary(str(config.paths.output_model_dir))
        logger.info(f"COLMAP binary model saved to: {config.paths.output_model_dir}")
        logger.info(f"-> COLMAP model written in {time.time() - step_start_time:.2f}s.")
    else:
        logger.warning("No dense points were generated. Skipping save.")

    logger.info(f"Total script execution time: {time.time() - total_start_time:.2f}s")
    logger.info("=== Summary ===")
    logger.info(f"Sampling strategy: {config.sampling.sampling_strategy}")
    logger.info(
        f"Edge weight: {config.sampling.edge_weight} (Random: {1 - config.sampling.edge_weight:.1%}, Gradient: {config.sampling.edge_weight:.1%})"
    )
    logger.info(f"Points per image: {config.sampling.num_points_per_image}")
    if all_dense_points:
        logger.info(f"Total points generated: {len(final_point_cloud)}")


if __name__ == "__main__":
    tyro.cli(main)
