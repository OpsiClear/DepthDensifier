import numpy as np
import torch
import pycolmap 
from moge.model.v2 import MoGeModel
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import time

# Import the refiner from your source directory
from depthdensifier.refiner import DepthRefiner, cp, device as refiner_device

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
PIPELINE_DOWNSAMPLE_FACTOR = 16

# Controls the final density of the point cloud, relative to the processing resolution.
# 1 = densest, 2 = half density, etc.
DOWNSAMPLE_DENSITY = 2

# --- Refiner Parameters ---
REFINER_CONFIG = {
    "lambda1": 1.0,
    "lambda2": 10.0,
    "verbose": 1,
}

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

        refinement_results = refiner.refine_depth(
            depth_map=moge_depth,
            normal_map=moge_normal,
            points3D=points3D_world,
            cam_from_world=image.cam_from_world().matrix(),
            K=camera.calibration_matrix()
        )
        if REFINER_CONFIG['verbose'] > 0:
            print(f"  - Depth Refinement: {time.time() - step_start_time:.2f}s")
        
        refined_depth = refinement_results['refined_depth']
        
        # --- Densify using the refined (and already downsampled) depth map ---
        step_start_time = time.time()
        h, w = refined_depth.shape
        pixels_y, pixels_x = np.mgrid[0:h:DOWNSAMPLE_DENSITY, 0:w:DOWNSAMPLE_DENSITY]
        valid_pixels = moge_mask[pixels_y, pixels_x].astype(bool)
        
        pixels_x, pixels_y = pixels_x[valid_pixels], pixels_y[valid_pixels]
        depth_values = refined_depth[pixels_y, pixels_x]
        
        points2D = np.stack([pixels_x, pixels_y], axis=-1)
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
        if refiner_device == 'cuda':
            cp.get_default_memory_pool().free_all_blocks()
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

        step_start_time = time.time()
        save_ply(OUTPUT_PLY, final_point_cloud)
        print(f"-> PLY file saved in {time.time() - step_start_time:.2f}s.")

        # --- Update the reconstruction object and save as COLMAP model ---
        step_start_time = time.time()
        print(f"Removing {rec.num_points3D()} old sparse points...")
        old_point3D_ids = list(rec.points3D.keys())
        for pid in old_point3D_ids:
            rec.delete_point3D(pid)
        print(f"-> Old points removed in {time.time() - step_start_time:.2f}s.")

        step_start_time = time.time()
        print(f"Adding {len(final_point_cloud)} new dense points...")
        for xyz in final_point_cloud:
            rec.add_point3D(xyz=xyz, track=pycolmap.Track(), color=np.array([128, 128, 128]))
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