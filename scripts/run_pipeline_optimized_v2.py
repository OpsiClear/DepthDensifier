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


@dataclass
class PathsConfig:
    """Configuration for input and output paths."""
    recon_path: Path = Path("data/360_v2/bicycle/sparse/0")
    image_dir: Path = Path("data/360_v2/bicycle/images")
    output_model_dir: Path = Path("results/360_v2/bicycle/sparse/0")


@dataclass
class MoGeConfig:
    """Configuration for the MoGe model."""
    checkpoint: Path = Path("models/moge/moge-2-vitl-normal/model.pt")


@dataclass
class ProcessingConfig:
    """Parameters for processing and densification."""
    pipeline_downsample_factor: int = 1
    """Factor to downsample images before processing."""
    downsample_density: int = 32
    """Controls final point cloud density (1=densest)."""
    batch_size: int = 4
    """Number of images to process in each GPU batch."""
    gpu_cache_clear_interval: int = 10
    """Clear GPU cache every N batches to prevent memory buildup."""


@dataclass
class ScriptConfig:
    """Main configuration for the densification script."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    moge: MoGeConfig = field(default_factory=MoGeConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    filtering: FloaterFilterConfig = field(default_factory=FloaterFilterConfig)


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


def process_batch(
    batch_data: list[tuple[pycolmap.Image, Path]],
    model: MoGeModel,
    rec: pycolmap.Reconstruction,
    refiner: DepthRefiner,
    config: ScriptConfig,
    sparse_points_cache: dict,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict]:
    """Process a batch of images through MoGe and refinement."""
    
    # Load and preprocess images
    batch_tensors = []
    batch_images = []
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
        
        # Densify
        pixels_y, pixels_x = np.mgrid[
            0:h:config.processing.downsample_density,
            0:w:config.processing.downsample_density,
        ]
        
        valid_pixels = refined_depth[pixels_y, pixels_x] > 0
        if not np.any(valid_pixels):
            continue
        
        pixels_x_valid = pixels_x[valid_pixels]
        pixels_y_valid = pixels_y[valid_pixels]
        
        # Get colors from the original tensor
        img_tensor_cpu = batch_tensors[idx].permute(1, 2, 0).numpy()
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


def main(config: ScriptConfig):
    total_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. MoGe inference will be slow.")
    
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
    logger.info(f"-> COLMAP reconstruction loaded in {time.time() - step_start_time:.2f}s.")
    
    # --- 3. Initialize Depth Refiner ---
    step_start_time = time.time()
    logger.info("Initializing Depth Refiner...")
    refiner_config = dataclasses.asdict(config.refiner)
    refiner = DepthRefiner(**refiner_config)
    logger.info(f"-> Depth Refiner initialized in {time.time() - step_start_time:.2f}s.")
    
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
    image_list = [(img, config.paths.image_dir / img.name) 
                  for img in rec.images.values() 
                  if img.has_pose and img.image_id in sparse_points_cache]
    
    num_batches = (len(image_list) + config.processing.batch_size - 1) // config.processing.batch_size
    
    logger.info(f"Processing {len(image_list)} images in {num_batches} batches...")
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * config.processing.batch_size
        end_idx = min(start_idx + config.processing.batch_size, len(image_list))
        batch_data = image_list[start_idx:end_idx]
        
        # Process batch
        batch_dense_points, batch_dense_colors, batch_dense_normals, batch_cached = process_batch(
            batch_data, model, rec, refiner, config, sparse_points_cache, device
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
        logger.info(f"Processed batch {batch_idx + 1}/{num_batches}, total points: {total_points}")
    
    logger.info(f"-> Image processing finished in {time.time() - processing_start_time:.2f}s.")
    
    # --- 6. Concatenate and filter points ---
    if all_dense_points:
        step_start_time = time.time()
        final_point_cloud = np.concatenate(all_dense_points, axis=0)
        final_colors = np.concatenate(all_dense_colors, axis=0)
        final_normals = np.concatenate(all_dense_normals, axis=0)
        logger.info(f"-> Point cloud concatenated in {time.time() - step_start_time:.2f}s.")
        
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


if __name__ == "__main__":
    tyro.cli(main)