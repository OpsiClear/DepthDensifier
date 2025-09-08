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
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue

# Import the refiner and floater filter from your source directory
from depthdensifier.depth_refiner import DepthRefiner, RefinerConfig
from depthdensifier.floater_filter import FloaterFilter, FloaterFilterConfig

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

    recon_path: Path = Path("data/bicycle/sparse/0")
    image_dir: Path = Path("data/bicycle/images")
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
    batch_size: int = 4
    """Number of images to process in each GPU batch."""
    gpu_cache_clear_interval: int = 10
    """Clear GPU cache every N images to prevent memory buildup."""
    prefetch_batches: int = 2
    """Number of batches to prefetch ahead of processing."""
    io_threads: int = 4
    """Number of threads for async I/O operations."""


# Use FloaterFilterConfig from the module instead of legacy config


@dataclass
class ScriptConfig:
    """Main configuration for the densification script."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    moge: MoGeConfig = field(default_factory=MoGeConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    filtering: FloaterFilterConfig = field(default_factory=FloaterFilterConfig)


def project_points(
    points3d: np.ndarray, image: pycolmap.Image, camera: pycolmap.Camera
) -> tuple[np.ndarray, np.ndarray]:
    """Projects 3D points to the image plane for a given camera."""
    # World to camera transformation
    cam_from_world = image.cam_from_world().matrix()
    points3d_h = np.hstack([points3d, np.ones((len(points3d), 1))])
    points_cam_h = (cam_from_world @ points3d_h.T).T

    points_cam = points_cam_h[:, :3]
    depths = points_cam[:, 2]

    # Camera to image plane projection
    points_cam_normalized = points_cam / (depths[:, np.newaxis] + 1e-8)

    K = camera.calibration_matrix()
    points2d_h = (K @ points_cam_normalized.T).T
    points2d = points2d_h[:, :2]
    return points2d, depths


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


def load_image_sync(
    image_path: Path, new_w: int, new_h: int
) -> tuple[torch.Tensor, Image.Image]:
    """Synchronous image loading and preprocessing."""
    pil_image = Image.open(image_path).convert("RGB")
    pil_image_rescaled = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Direct tensor creation
    img_tensor = torch.from_numpy(np.array(pil_image_rescaled)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)

    return img_tensor, pil_image_rescaled


class AsyncImageLoader:
    """Async image loader with prefetching capabilities."""

    def __init__(self, num_threads: int = 4, prefetch_size: int = 8):
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.prefetch_queue: Queue = Queue(maxsize=prefetch_size)
        self.prefetch_thread = None
        self.stop_prefetching = threading.Event()

    def start_prefetching(
        self,
        image_batches: list[list[tuple[pycolmap.Image, Path]]],
        new_w: int,
        new_h: int,
    ):
        """Start prefetching images in background thread."""

        def prefetch_worker():
            for batch_idx, batch_data in enumerate(image_batches):
                if self.stop_prefetching.is_set():
                    break

                # Load batch of images
                futures = []
                for image, image_path in batch_data:
                    future = self.executor.submit(
                        load_image_sync, image_path, new_w, new_h
                    )
                    futures.append((image, future))

                # Collect results
                batch_tensors = []
                batch_pil_images = []
                batch_images = []

                for image, future in futures:
                    try:
                        img_tensor, pil_image_rescaled = future.result()
                        batch_tensors.append(img_tensor)
                        batch_pil_images.append(pil_image_rescaled)
                        batch_images.append(image)
                    except Exception as e:
                        logger.warning(f"Failed to load image {image.name}: {e}")
                        continue

                if batch_tensors:
                    batch_result = {
                        "batch_idx": batch_idx,
                        "batch_images": batch_images,
                        "batch_tensors": torch.stack(batch_tensors),
                        "batch_pil_images": batch_pil_images,
                    }
                    self.prefetch_queue.put(batch_result)

        self.prefetch_thread = threading.Thread(target=prefetch_worker)
        self.prefetch_thread.start()

    def get_batch(self, timeout: float = 30.0) -> dict | None:
        """Get next prefetched batch."""
        try:
            return self.prefetch_queue.get(timeout=timeout)
        except Exception:
            return None

    def stop(self):
        """Stop prefetching and cleanup."""
        self.stop_prefetching.set()
        if self.prefetch_thread:
            self.prefetch_thread.join()
        self.executor.shutdown(wait=True)


def process_batch(
    batch_images: list[pycolmap.Image],
    batch_tensors: torch.Tensor,
    model: MoGeModel,
    rec: pycolmap.Reconstruction,
    config: ScriptConfig,
    refiner: DepthRefiner,
    device: torch.device,
    sparse_points_cache: dict[int, np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict]:
    """Process a batch of images through MoGe and refinement."""

    # Run batch inference
    with torch.no_grad():
        batch_outputs = model.infer(batch_tensors)

    batch_dense_points = []
    batch_dense_colors = []
    batch_dense_normals = []
    cached_data = {}

    # Process each image in the batch
    for idx, image in enumerate(batch_images):
        # Extract outputs for this image
        moge_depth = batch_outputs["depth"][idx].cpu().numpy()
        moge_normal = batch_outputs["normal"][idx].cpu().numpy()
        moge_mask = batch_outputs["mask"][idx].cpu().numpy()

        # Get cached sparse points for refinement
        if image.image_id not in sparse_points_cache:
            continue  # Skip images with no sparse points

        points3D_world = sparse_points_cache[image.image_id]

        # Setup camera at processing resolution
        camera = rec.cameras[image.camera_id]
        h, w = moge_depth.shape
        camera.rescale(new_width=w, new_height=h)

        # Handle sky regions BEFORE refinement to prevent inf corruption
        # Identify sky pixels (MoGe sets them to inf)
        sky_mask = np.isinf(moge_depth) | (moge_depth > 1000)

        # Create a copy of depth for refinement, replacing inf with large finite values
        depth_for_refinement = moge_depth.copy()
        if np.any(sky_mask):
            # Use a large but finite placeholder for sky during refinement
            # This prevents inf from corrupting the interpolation
            temp_sky_depth = 1000.0  # Large but finite
            depth_for_refinement[sky_mask] = temp_sky_depth

            if config.refiner.verbose > 1:
                sky_pixel_count = np.sum(sky_mask)
                logger.debug(
                    f"  - Pre-refinement: {sky_pixel_count} sky pixels temporarily set to {temp_sky_depth}"
                )

        # Refine depth with finite values only
        cam_from_world_mat = image.cam_from_world().matrix()[:3, :]
        K_mat = camera.calibration_matrix()
        refinement_results = refiner.refine_depth(
            depth_map=depth_for_refinement,
            normal_map=moge_normal,
            points3D=points3D_world,
            cam_from_world=cam_from_world_mat,
            K=K_mat,
            mask=moge_mask,
        )

        refined_depth = refinement_results["refined_depth"]

        # Now properly handle sky regions with 2x max depth
        # Calculate maximum valid (non-sky) depth in the view
        valid_depth_mask = moge_mask & ~sky_mask & (refined_depth > 0)
        if np.any(valid_depth_mask) and np.any(sky_mask):
            max_valid_depth = np.max(refined_depth[valid_depth_mask])
            sky_depth_value = 2.0 * max_valid_depth

            # Apply final sky depth (2x max of valid depths)
            refined_depth[sky_mask] = sky_depth_value

            if config.refiner.verbose > 1:
                logger.debug(
                    f"  - Post-refinement: Sky pixels set to {sky_depth_value:.2f} (2x max depth {max_valid_depth:.2f})"
                )

        # Set invalid regions to 0
        refined_depth[~moge_mask] = 0

        # Cache for filtering
        cached_data[image.image_id] = {
            "refined_depth": refined_depth,
            "camera": camera,
            "image": image,
        }

        # Densify using vectorized operations
        pixels_y, pixels_x = np.mgrid[
            0 : h : config.processing.downsample_density,
            0 : w : config.processing.downsample_density,
        ]

        valid_pixels = refined_depth[pixels_y, pixels_x] > 0
        if not np.any(valid_pixels):
            continue

        pixels_x_valid, pixels_y_valid = pixels_x[valid_pixels], pixels_y[valid_pixels]

        # Get colors and normals
        # Note: we need the original image tensor to get colors
        img_tensor_cpu = batch_tensors[idx].cpu().permute(1, 2, 0).numpy()
        img_np = (img_tensor_cpu * 255).astype(np.uint8)
        colors = img_np[pixels_y_valid, pixels_x_valid]
        normals = moge_normal[pixels_y_valid, pixels_x_valid]

        # Unproject to 3D
        depth_values = refined_depth[pixels_y_valid, pixels_x_valid]
        points2D = np.stack([pixels_x_valid, pixels_y_valid], axis=-1)
        points3D_camera = unproject_points(points2D, depth_values, camera)
        points3D_world = image.cam_from_world().inverse() * points3D_camera

        batch_dense_points.append(points3D_world)
        batch_dense_colors.append(colors)
        batch_dense_normals.append(normals)

    return batch_dense_points, batch_dense_colors, batch_dense_normals, cached_data


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

    # --- 4. Pre-compute sparse points cache for all images ---
    step_start_time = time.time()
    logger.info("Pre-computing sparse points cache...")
    image_list = [img for img in rec.images.values() if img.has_pose]

    sparse_points_cache: dict[int, np.ndarray] = {}
    cache_stats = {
        "total_images": len(image_list),
        "cached_images": 0,
        "total_sparse_points": 0,
    }

    for image in image_list:
        point3D_ids = [p.point3D_id for p in image.points2D if p.has_point3D()]
        if len(point3D_ids) > 0:
            points3D_world = np.array([rec.points3D[pid].xyz for pid in point3D_ids])
            sparse_points_cache[image.image_id] = points3D_world
            cache_stats["cached_images"] += 1
            cache_stats["total_sparse_points"] += len(points3D_world)

    logger.info(
        f"-> Sparse points cache created: {cache_stats['cached_images']}/{cache_stats['total_images']} "
        f"images cached with {cache_stats['total_sparse_points']} total sparse points "
        f"in {time.time() - step_start_time:.2f}s."
    )

    # --- Pre-compute camera matrices for filtering optimization ---
    step_start_time = time.time()
    logger.info("Pre-computing camera matrices...")
    camera_matrices_cache: dict[int, dict] = {}

    for image in image_list:
        camera = rec.cameras[image.camera_id]

        # Pre-compute matrices for optimization
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

    # --- 5. Pre-allocate arrays for better performance ---
    # Estimate total points (with 50% overhead)
    sample_image = Image.open(config.paths.image_dir / image_list[0].name)
    w_orig, h_orig = sample_image.size
    new_w = w_orig // config.processing.pipeline_downsample_factor
    new_h = h_orig // config.processing.pipeline_downsample_factor
    points_per_image = (new_h // config.processing.downsample_density) * (
        new_w // config.processing.downsample_density
    )
    estimated_total_points = int(points_per_image * len(image_list) * 1.5)

    logger.info(
        f"Pre-allocating arrays for approximately {estimated_total_points} points..."
    )
    all_dense_points = np.zeros((estimated_total_points, 3), dtype=np.float32)
    all_dense_colors = np.zeros((estimated_total_points, 3), dtype=np.uint8)
    all_dense_normals = np.zeros((estimated_total_points, 3), dtype=np.float32)
    point_idx = 0

    # --- 6. Setup async image loading with prefetching ---
    processing_start_time = time.time()
    cached_refinement_data = {}
    batch_size = config.processing.batch_size

    # Prepare batch data for async loading
    image_batches = []
    for batch_start in range(0, len(image_list), batch_size):
        batch_end = min(batch_start + batch_size, len(image_list))
        batch_images = image_list[batch_start:batch_end]
        batch_data = [
            (image, config.paths.image_dir / image.name) for image in batch_images
        ]
        image_batches.append(batch_data)

    # Initialize async loader
    async_loader = AsyncImageLoader(
        num_threads=config.processing.io_threads,
        prefetch_size=config.processing.prefetch_batches,
    )

    logger.info(
        f"Starting async image loading with {config.processing.io_threads} threads, "
        f"prefetching {config.processing.prefetch_batches} batches ahead..."
    )

    # Start prefetching
    async_loader.start_prefetching(image_batches, new_w, new_h)

    # --- 7. Process images with async I/O ---
    try:
        for batch_idx in tqdm(range(len(image_batches)), desc="Processing batches"):
            # Get prefetched batch
            batch_data = async_loader.get_batch()
            if batch_data is None:
                logger.warning(f"Failed to get batch {batch_idx}, skipping...")
                continue

            batch_images = batch_data["batch_images"]
            batch_tensor = batch_data["batch_tensors"].to(device)

            # Process batch
            batch_points, batch_colors, batch_normals, batch_cached = process_batch(
                batch_images,
                batch_tensor,
                model,
                rec,
                config,
                refiner,
                device,
                sparse_points_cache,
            )

            # Store results in pre-allocated arrays
            for points, colors, normals in zip(
                batch_points, batch_colors, batch_normals
            ):
                n_points = len(points)
                if point_idx + n_points > len(all_dense_points):
                    # Resize if needed
                    logger.warning("Resizing pre-allocated arrays...")
                    new_size = int(len(all_dense_points) * 1.5)
                    all_dense_points = np.resize(all_dense_points, (new_size, 3))
                    all_dense_colors = np.resize(all_dense_colors, (new_size, 3))
                    all_dense_normals = np.resize(all_dense_normals, (new_size, 3))

                all_dense_points[point_idx : point_idx + n_points] = points
                all_dense_colors[point_idx : point_idx + n_points] = colors
                all_dense_normals[point_idx : point_idx + n_points] = normals
                point_idx += n_points

            cached_refinement_data.update(batch_cached)

            # Clear GPU cache periodically
            if batch_idx % config.processing.gpu_cache_clear_interval == 0:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            logger.info(f"Processed batch {batch_idx + 1}, total points: {point_idx}")

    finally:
        # Cleanup async loader
        async_loader.stop()
        logger.info("Async image loader stopped.")

    # Trim arrays to actual size
    final_point_cloud = all_dense_points[:point_idx]
    final_colors = all_dense_colors[:point_idx]
    final_normals = all_dense_normals[:point_idx]

    logger.info(
        f"-> Image processing finished in {time.time() - processing_start_time:.2f}s."
    )

    # --- 6. Optimized filtering with FloaterFilter module ---
    if point_idx > 0:
        logger.info("--- Filtering point cloud for geometric consistency ---")
        filter_start_time = time.time()

        # Initialize floater filter with FloaterFilterConfig
        floater_filter = FloaterFilter(config.filtering)

        # Prepare camera data using the filter's helper method
        camera_data = floater_filter.prepare_camera_data(
            cached_refinement_data, camera_matrices_cache
        )

        # Filter points using the module
        points_to_keep_mask, num_removed = floater_filter.filter_points(
            final_point_cloud, final_normals, camera_data, show_progress=True
        )

        # Apply the filtering mask
        final_point_cloud = final_point_cloud[points_to_keep_mask]
        final_colors = final_colors[points_to_keep_mask]

        logger.info(
            f"-> Filtering removed {num_removed} points ({num_removed / point_idx * 100:.2f}%)"
        )
        logger.info(f"-> Filtering finished in {time.time() - filter_start_time:.2f}s.")

        # --- 7. Vectorized point addition to reconstruction ---
        step_start_time = time.time()
        logger.info(f"Adding {len(final_point_cloud)} new dense points...")

        # Batch add points (more efficient than individual adds)
        for i in range(0, len(final_point_cloud), 1000):
            batch_end = min(i + 1000, len(final_point_cloud))
            for j in range(i, batch_end):
                rec.add_point3D(
                    xyz=final_point_cloud[j],
                    track=pycolmap.Track(),
                    color=final_colors[j],
                )

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


if __name__ == "__main__":
    tyro.cli(main)
