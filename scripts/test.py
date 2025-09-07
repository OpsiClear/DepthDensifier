import numpy as np
import torch
import pycolmap 
from moge.model.v2 import MoGeModel
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import time

# Import the refiner from your source directory
from depthdensifier.depth_refiner import DepthRefiner
from depthdensifier.initilizer import compute_image_gradients_gpu

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# --- Input Paths ---
RECON_PATH = Path("data/360_v2/bicycle/sparse/0")
IMAGE_DIR = Path("data/360_v2/bicycle/images")

# --- Output Path ---
OUTPUT_PLY = Path("results/dense_point_cloud_refined.ply")
OUTPUT_MODEL_DIR = Path("results/0")

# --- MoGe Configuration ---
MOGE_CHECKPOINT = "models/moge/moge-2-vitl-normal/model.pt" 

# --- Processing & Densification Parameters ---
# Factor by which to downsample images before MoGe inference and refinement.
# A larger factor is MUCH faster and uses less memory. (e.g., 4 or 8).
PIPELINE_DOWNSAMPLE_FACTOR = 1

# Controls the final density of the point cloud, relative to the processing resolution.
# 1 = densest, 2 = half density, etc.
DOWNSAMPLE_DENSITY = 32

# --- Refiner Parameters ---
REFINER_CONFIG = {
    "min_correspondences": 50,
    "edge_margin": 10,
    "robust": True,
    "outlier_threshold": 2.5,
    "use_fp16": True,
    "skip_smoothing": False,
    "adaptive_correspondences": True,
    "verbose": 1,
}


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
    points3D_camera = np.stack([
        x_normalized * depth,
        y_normalized * depth,
        depth
    ], axis=-1)
    return points3D_camera

def save_ply(path, points):
    """Saves a point cloud to a PLY file."""
    if not points.any():
        print("Warning: No points to save.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
"""
    with open(path, 'w') as f:
        f.write(header)
        np.savetxt(f, points, fmt='%.6f')
    print(f"Dense point cloud saved to {path}")

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
def main():
    total_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. MoGe inference will be slow.")
        
    # --- 1. Load MoGe Model ---
    step_start_time = time.time()
    print(f"Loading MoGe model ({MOGE_CHECKPOINT})...")
    model = MoGeModel.from_pretrained(MOGE_CHECKPOINT).to(device)
    model.eval()
    print(f"-> MoGe model loaded in {time.time() - step_start_time:.2f}s.")

    # --- 2. Load COLMAP Reconstruction ---
    step_start_time = time.time()
    print(f"Loading COLMAP reconstruction from {RECON_PATH}...")
    rec = pycolmap.Reconstruction(RECON_PATH)
    print(f"Loaded model with {rec.num_reg_images()} images and {rec.num_points3D()} sparse points.")
    print(f"-> COLMAP reconstruction loaded in {time.time() - step_start_time:.2f}s.")

    # --- 3. Initialize the Depth Refiner ---
    step_start_time = time.time()
    print("Initializing Depth Refiner...")
    refiner = DepthRefiner(**REFINER_CONFIG)
    print(f"-> Depth Refiner initialized in {time.time() - step_start_time:.2f}s.")

    # --- 4. Process each image: Infer, Refine, and Densify ---
    processing_start_time = time.time()
    all_dense_points = []
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
        if REFINER_CONFIG['verbose'] > 0:
            print(f"  - Get sparse points: {time.time() - step_start_time:.2f}s")

        # --- Load and Downsample Image FIRST ---
        step_start_time = time.time()
        image_path = IMAGE_DIR / image.name
        pil_image = Image.open(image_path).convert("RGB")
        
        w_orig, h_orig = pil_image.size
        new_w = w_orig // PIPELINE_DOWNSAMPLE_FACTOR
        new_h = h_orig // PIPELINE_DOWNSAMPLE_FACTOR
        
        pil_image_rescaled = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        img_tensor = torch.from_numpy(np.array(pil_image_rescaled)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(device)
        if REFINER_CONFIG['verbose'] > 0:
            print(f"  - Image Load & Downsample: {time.time() - step_start_time:.2f}s")

        # --- Run MoGe Inference on the smaller image ---
        step_start_time = time.time()
        with torch.no_grad():
            moge_output = model.infer(img_tensor)
        if REFINER_CONFIG['verbose'] > 0:
            print(f"  - MoGe Inference: {time.time() - step_start_time:.2f}s")
        
        moge_depth = moge_output['depth'].squeeze(0).cpu().numpy()
        moge_normal = moge_output['normal'].squeeze(0).cpu().numpy()
        moge_mask = moge_output['mask'].squeeze(0).cpu().numpy()

        # --- Refine the depth map at the processing resolution ---
        step_start_time = time.time()
        camera = rec.cameras[image.camera_id]
        camera.rescale(new_width=new_w, new_height=new_h)
        if REFINER_CONFIG['verbose'] > 0:
            print(f"\n--- Refining depth for {image.name} (ID: {image.image_id}) ---")

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
        if REFINER_CONFIG['verbose'] > 0:
            print(f"  - Depth Refinement: {time.time() - step_start_time:.2f}s")
        
        refined_depth = refinement_results['refined_depth']
        
        # Apply the MoGe mask to the refined depth map before caching and densification.
        # This ensures we only work with depths from the object of interest going forward.
        refined_depth[~moge_mask] = 0
        
        # Cache data for filtering step
        cached_refinement_data[image.image_id] = {
            'refined_depth': refined_depth,
            'camera': camera,
            'image': image
        }

        # --- Densify using the refined (and already downsampled) depth map ---
        step_start_time = time.time()
        h, w = refined_depth.shape
        pixels_y, pixels_x = np.mgrid[0:h:DOWNSAMPLE_DENSITY, 0:w:DOWNSAMPLE_DENSITY]
        
        # The refined depth map already has the MoGe mask applied.
        # We create points only where the depth is non-zero.
        valid_pixels = refined_depth[pixels_y, pixels_x] > 0
        
        pixels_x_valid, pixels_y_valid = pixels_x[valid_pixels], pixels_y[valid_pixels]
        
        # --- [DEBUG] Unproject unrefined depth for comparison ---
        depth_values_unrefined = moge_depth[pixels_y_valid, pixels_x_valid]
        points2D_unrefined = np.stack([pixels_x_valid, pixels_y_valid], axis=-1)
        points3D_camera_unrefined = unproject_points(points2D_unrefined, depth_values_unrefined, camera)
        points3D_world_unrefined = image.cam_from_world().inverse() * points3D_camera_unrefined
        all_unrefined_points.append(points3D_world_unrefined)
        
        depth_values = refined_depth[pixels_y_valid, pixels_x_valid]
        
        points2D = np.stack([pixels_x_valid, pixels_y_valid], axis=-1)
        points3D_camera = unproject_points(points2D, depth_values, camera)
        points3D_world = image.cam_from_world().inverse() * points3D_camera
        if REFINER_CONFIG['verbose'] > 0:
            print(f"  - Densification calculation: {time.time() - step_start_time:.2f}s")

        append_start_time = time.time()
        all_dense_points.append(points3D_world)
        if REFINER_CONFIG['verbose'] > 0:
            print(f"  - Appending points to list: {time.time() - append_start_time:.2f}s")

        num_points += len(points3D_world)
        print(f"number of dense points: {num_points}")
        
        # --- Explicitly clean up GPU memory to prevent slowdown ---
        cleanup_start_time = time.time()
        del img_tensor, moge_output, refinement_results
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if REFINER_CONFIG['verbose'] > 0:
            print(f"  - GPU Memory Cleanup: {time.time() - cleanup_start_time:.2f}s")

        if REFINER_CONFIG['verbose'] > 0:
            print(f"-> Processed image ({image.name}) in {time.time() - per_image_start_time:.2f}s")

    print(f"-> Image processing loop finished in {time.time() - processing_start_time:.2f}s.")

    # --- 5. Save the final results ---
    saving_start_time = time.time()
    if all_dense_points:
        step_start_time = time.time()
        final_point_cloud = np.concatenate(all_dense_points, axis=0)
        print(f"-> Point cloud concatenated in {time.time() - step_start_time:.2f}s.")

        # --- Filter point cloud based on multi-view consistency ---
        print("\n--- Filtering point cloud for geometric consistency ---")
        filter_start_time = time.time()
        
        floater_votes = np.zeros(len(final_point_cloud), dtype=int)
        VOTE_THRESHOLD = 5 # A point is removed if 2+ cameras see it as a floater.

        for data in tqdm(cached_refinement_data.values(), desc="Filtering Points"):
            refined_depth = data['refined_depth']
            h, w = refined_depth.shape
            
            # Project points into the current view
            points2d, depths = project_points(final_point_cloud, data['image'], data['camera'])
            
            u, v = points2d[:, 0], points2d[:, 1]
            
            # Create a mask for points that project inside the image bounds
            mask_in_bounds = (
                (u >= 0) & (u < w) & (v >= 0) & (v < h) & (depths > 0)
            )
            
            if not np.any(mask_in_bounds):
                continue
                
            # Get integer coordinates for depth lookup
            u_valid = u[mask_in_bounds].astype(int)
            v_valid = v[mask_in_bounds].astype(int)
            
            projected_depths_valid = depths[mask_in_bounds]
            refined_depths_at_projections = refined_depth[v_valid, u_valid]
            
            # Create a mask for where the lookup is valid (non-zero depth)
            valid_lookup_mask = refined_depths_at_projections > 0
            
            # A point is a "floater" if its projected depth is significantly
            # LESS than the depth map's value (i.e., it's between the camera and the surface).
            inconsistent_mask = (
                projected_depths_valid[valid_lookup_mask] < 0.7 * refined_depths_at_projections[valid_lookup_mask]
            )
            
            # Get the original indices of inconsistent points and increment their vote count
            original_indices_in_bounds = np.where(mask_in_bounds)[0]
            indices_with_valid_lookup = original_indices_in_bounds[valid_lookup_mask]
            inconsistent_indices = indices_with_valid_lookup[inconsistent_mask]
            
            floater_votes[inconsistent_indices] += 1
            
        points_to_keep_mask = floater_votes < VOTE_THRESHOLD
        final_point_cloud = final_point_cloud[points_to_keep_mask]
        num_removed = np.sum(~points_to_keep_mask)
        print(f"-> Filtering removed {num_removed} points ({num_removed / len(points_to_keep_mask) * 100:.2f}%)")
        print(f"-> Filtering finished in {time.time() - filter_start_time:.2f}s.")


        step_start_time = time.time()
        save_ply(OUTPUT_PLY, final_point_cloud)
        print(f"-> PLY file saved in {time.time() - step_start_time:.2f}s.")

        # --- Update the reconstruction object and save as COLMAP model ---
        step_start_time = time.time()
        print(f"Removing {rec.num_points3D()} old sparse points...")
        old_point3D_ids = list(rec.points3D.keys())
        for pid in old_point3D_ids:
            rec.points3D[pid].color = np.array([255, 0, 0])
        print(f"-> Old points removed in {time.time() - step_start_time:.2f}s.")
        
        # # --- [DEBUG] Add unrefined points (green) ---
        # if all_unrefined_points:
        #     unrefined_point_cloud = np.concatenate(all_unrefined_points, axis=0)
        #     print(f"Adding {len(unrefined_point_cloud)} new unrefined (green) points...")
        #     for xyz in unrefined_point_cloud:
        #         rec.add_point3D(xyz=xyz, track=pycolmap.Track(), color=np.array([0, 255, 0]))

        step_start_time = time.time()
        print(f"Adding {len(final_point_cloud)} new dense points...")
        for xyz in final_point_cloud:
            rec.add_point3D(xyz=xyz, track=pycolmap.Track(), color=np.array([0, 0, 0]))
        print(f"-> New points added in {time.time() - step_start_time:.2f}s.")

        step_start_time = time.time()
        OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        rec.write_binary(str(OUTPUT_MODEL_DIR))
        print(f"COLMAP binary model saved to: {OUTPUT_MODEL_DIR}")
        print(f"-> COLMAP model written in {time.time() - step_start_time:.2f}s.")
    else:
        print("No dense points were generated. Skipping save.")

    print(f"-> Saving finished in {time.time() - saving_start_time:.2f}s.")
    print(f"\nTotal script execution time: {time.time() - total_start_time:.2f}s")

if __name__ == "__main__":
    main()