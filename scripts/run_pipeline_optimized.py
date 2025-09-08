"""
Optimized pipeline with batch processing and performance improvements.
"""

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

# Import the refiner
from src.depthdensifier.depth_refiner import DepthRefiner, RefinerConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


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
    downsample_density: int = 32
    batch_size: int = 4  # NEW: Process multiple images at once
    use_fp16: bool = True  # NEW: Use half precision for MoGe
    preload_images: bool = True  # NEW: Preload images in parallel


@dataclass
class FilteringConfig:
    """Parameters for multi-view consistency filtering."""
    vote_threshold: int = 5
    depth_threshold: float = 0.7


@dataclass
class ScriptConfig:
    """Main configuration for the densification script."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    moge: MoGeConfig = field(default_factory=MoGeConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)


def project_points_vectorized(
    points3d: np.ndarray, images: list, cameras: list
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Vectorized projection of 3D points to multiple image planes."""
    results = []
    for image, camera in zip(images, cameras):
        cam_from_world = image.cam_from_world().matrix()
        points3d_h = np.hstack([points3d, np.ones((len(points3d), 1))])
        points_cam_h = (cam_from_world @ points3d_h.T).T
        
        points_cam = points_cam_h[:, :3]
        depths = points_cam[:, 2]
        
        points_cam_normalized = points_cam / (depths[:, np.newaxis] + 1e-8)
        K = camera.calibration_matrix()
        points2d_h = (K @ points_cam_normalized.T).T
        points2d = points2d_h[:, :2]
        results.append((points2d, depths))
    return results


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


def load_image_batch(image_paths, downsample_factor, device, dtype=torch.float32):
    """Load and preprocess a batch of images in parallel."""
    def load_single(path):
        pil_img = Image.open(path).convert("RGB")
        w_orig, h_orig = pil_img.size
        new_w = w_orig // downsample_factor
        new_h = h_orig // downsample_factor
        pil_img_rescaled = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return pil_img_rescaled, new_w, new_h
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(load_single, image_paths))
    
    # Convert to batch tensor
    images_rescaled = []
    tensors = []
    for pil_img, new_w, new_h in results:
        images_rescaled.append(pil_img)
        img_array = np.array(pil_img)
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        tensors.append(tensor)
    
    batch_tensor = torch.stack(tensors).to(device, dtype=dtype)
    return batch_tensor, images_rescaled, results[0][1], results[0][2]  # new_w, new_h


def process_batch(
    batch_images, batch_data, model, refiner, rec, config, device
):
    """Process a batch of images together."""
    batch_size = len(batch_images)
    
    # Prepare batch data
    image_paths = [config.paths.image_dir / img.name for img in batch_images]
    
    # Load images in parallel
    dtype = torch.float16 if config.processing.use_fp16 else torch.float32
    img_batch_tensor, pil_images_rescaled, new_w, new_h = load_image_batch(
        image_paths, config.processing.pipeline_downsample_factor, device, dtype
    )
    
    # Run MoGe inference on batch
    with torch.no_grad():
        if config.processing.use_fp16:
            with torch.cuda.amp.autocast():
                moge_outputs = model.infer(img_batch_tensor)
        else:
            moge_outputs = model.infer(img_batch_tensor)
    
    # Process each image in the batch
    batch_results = []
    for i in range(batch_size):
        image = batch_images[i]
        
        # Extract individual outputs
        moge_depth = moge_outputs["depth"][i].cpu().numpy()
        moge_normal = moge_outputs["normal"][i].cpu().numpy()
        moge_mask = moge_outputs["mask"][i].cpu().numpy()
        
        # Get sparse points
        point3D_ids = [p.point3D_id for p in image.points2D if p.has_point3D()]
        if len(point3D_ids) == 0:
            batch_results.append(None)
            continue
        
        points3D_world = np.array([rec.points3D[pid].xyz for pid in point3D_ids])
        
        # Setup camera
        camera = rec.cameras[image.camera_id]
        camera.rescale(new_width=new_w, new_height=new_h)
        
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
        
        # Densify
        h, w = refined_depth.shape
        pixels_y, pixels_x = np.mgrid[
            0 : h : config.processing.downsample_density,
            0 : w : config.processing.downsample_density,
        ]
        
        valid_pixels = refined_depth[pixels_y, pixels_x] > 0
        pixels_x_valid = pixels_x[valid_pixels]
        pixels_y_valid = pixels_y[valid_pixels]
        
        # Get colors and normals
        img_np_rescaled = np.array(pil_images_rescaled[i])
        colors = img_np_rescaled[pixels_y_valid, pixels_x_valid]
        normals = moge_normal[pixels_y_valid, pixels_x_valid]
        
        # Unproject to 3D
        depth_values = refined_depth[pixels_y_valid, pixels_x_valid]
        points2D = np.stack([pixels_x_valid, pixels_y_valid], axis=-1)
        points3D_camera = unproject_points(points2D, depth_values, camera)
        points3D_world = image.cam_from_world().inverse() * points3D_camera
        
        batch_results.append({
            "points": points3D_world,
            "colors": colors,
            "normals": normals,
            "refined_depth": refined_depth,
            "camera": camera,
            "image": image,
        })
    
    # Clean up GPU memory
    del img_batch_tensor, moge_outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return batch_results


def filter_points_optimized(
    points: np.ndarray,
    normals: np.ndarray,
    cached_data: dict,
    config: FilteringConfig
) -> np.ndarray:
    """Optimized multi-view filtering using vectorization."""
    num_points = len(points)
    floater_votes = np.zeros(num_points, dtype=np.int8)  # Use int8 to save memory
    
    # Pre-compute camera centers
    cam_centers = {}
    for img_id, data in cached_data.items():
        cam_centers[img_id] = data["image"].projection_center()
    
    for img_id, data in tqdm(cached_data.items(), desc="Filtering Points", leave=False):
        refined_depth = data["refined_depth"]
        h, w = refined_depth.shape
        cam_center = cam_centers[img_id]
        
        # Vectorized projection
        cam_from_world = data["image"].cam_from_world().matrix()
        points_h = np.hstack([points, np.ones((num_points, 1))])
        points_cam_h = (cam_from_world @ points_h.T).T
        points_cam = points_cam_h[:, :3]
        depths = points_cam[:, 2]
        
        # Early filtering: skip points behind camera
        front_mask = depths > 0
        if not np.any(front_mask):
            continue
        
        # Project to image plane
        points_cam_normalized = points_cam[front_mask] / depths[front_mask, np.newaxis]
        K = data["camera"].calibration_matrix()
        points2d_h = (K @ points_cam_normalized.T).T
        u, v = points2d_h[:, 0], points2d_h[:, 1]
        
        # Grazing angle check (vectorized)
        viewing_dirs = points[front_mask] - cam_center
        viewing_dirs /= np.linalg.norm(viewing_dirs, axis=1)[:, np.newaxis]
        dot_products = np.sum(normals[front_mask] * -viewing_dirs, axis=1)
        not_grazing = dot_products > 0.052
        
        # Bounds check
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h) & not_grazing
        if not np.any(in_bounds):
            continue
        
        # Depth consistency check
        u_valid = u[in_bounds].astype(np.int32)
        v_valid = v[in_bounds].astype(np.int32)
        projected_depths = depths[front_mask][in_bounds]
        refined_depths = refined_depth[v_valid, u_valid]
        
        valid_lookup = refined_depths > 0
        if np.any(valid_lookup):
            inconsistent = (
                projected_depths[valid_lookup] 
                < config.depth_threshold * refined_depths[valid_lookup]
            )
            
            # Update votes
            front_indices = np.where(front_mask)[0]
            in_bounds_indices = front_indices[in_bounds]
            valid_indices = in_bounds_indices[valid_lookup]
            floater_votes[valid_indices[inconsistent]] += 1
    
    return floater_votes < config.vote_threshold


def main(config: ScriptConfig):
    total_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load MoGe Model with FP16 support
    logger.info(f"Loading MoGe model...")
    model = MoGeModel.from_pretrained(config.moge.checkpoint).to(device)
    model.eval()
    if config.processing.use_fp16:
        model = model.half()
    
    # Load COLMAP Reconstruction
    logger.info(f"Loading COLMAP reconstruction...")
    rec = pycolmap.Reconstruction(config.paths.recon_path)
    logger.info(f"Loaded {rec.num_reg_images()} images, {rec.num_points3D()} points")
    
    # Initialize Depth Refiner
    refiner_config = dataclasses.asdict(config.refiner)
    refiner = DepthRefiner(**refiner_config)
    
    # Process images in batches
    all_dense_points = []
    all_dense_colors = []
    all_dense_normals = []
    cached_refinement_data = {}
    
    image_list = [img for img in rec.images.values() if img.has_pose]
    num_batches = (len(image_list) + config.processing.batch_size - 1) // config.processing.batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing Batches"):
        start_idx = batch_idx * config.processing.batch_size
        end_idx = min(start_idx + config.processing.batch_size, len(image_list))
        batch_images = image_list[start_idx:end_idx]
        
        # Process batch
        batch_results = process_batch(
            batch_images, None, model, refiner, rec, config, device
        )
        
        # Collect results
        for result in batch_results:
            if result is not None:
                all_dense_points.append(result["points"])
                all_dense_colors.append(result["colors"])
                all_dense_normals.append(result["normals"])
                cached_refinement_data[result["image"].image_id] = {
                    "refined_depth": result["refined_depth"],
                    "camera": result["camera"],
                    "image": result["image"],
                }
    
    # Combine all points
    if all_dense_points:
        final_point_cloud = np.concatenate(all_dense_points, axis=0)
        final_colors = np.concatenate(all_dense_colors, axis=0)
        final_normals = np.concatenate(all_dense_normals, axis=0)
        
        logger.info(f"Total points before filtering: {len(final_point_cloud)}")
        
        # Optimized filtering
        logger.info("Filtering points for consistency...")
        keep_mask = filter_points_optimized(
            final_point_cloud, final_normals, cached_refinement_data, config.filtering
        )
        
        final_point_cloud = final_point_cloud[keep_mask]
        final_colors = final_colors[keep_mask]
        
        logger.info(f"Points after filtering: {len(final_point_cloud)}")
        
        # Save results
        logger.info("Saving COLMAP model...")
        for i in range(len(final_point_cloud)):
            xyz = final_point_cloud[i]
            rgb = final_colors[i]
            rec.add_point3D(xyz=xyz, track=pycolmap.Track(), color=rgb)
        
        config.paths.output_model_dir.mkdir(parents=True, exist_ok=True)
        rec.write_binary(str(config.paths.output_model_dir))
        logger.info(f"Saved to: {config.paths.output_model_dir}")
    
    logger.info(f"Total time: {time.time() - total_start_time:.2f}s")


if __name__ == "__main__":
    tyro.cli(main)