import numpy as np
import pycolmap

try:
    import cupy as cp
    print("Using GPU acceleration (CuPy) for SimpleRefiner.")
    CUPY_AVAILABLE = True
except ImportError:
    print("CuPy not available, SimpleRefiner will use CPU (NumPy).")
    cp = np # Fallback to numpy if cupy is not installed
    CUPY_AVAILABLE = False


class SimpleRefiner:
    """
    A simple, fast refiner to find the optimal scale for a depth map using CuPy.
    It aligns a monocular depth map to sparse COLMAP points by finding a single
    scale factor that minimizes the log-depth difference via linear least-squares.
    """
    def __init__(self, verbose: int = 1):
        self.verbose = verbose

    def refine(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        points3D_world: np.ndarray,
        image: pycolmap.Image,
        camera: pycolmap.Camera,
    ) -> dict:
        """
        Calculates the optimal scale using GPU (CuPy) and applies it.

        Args:
            depth_map: The HxW monocular depth map from MoGe.
            mask: The HxW validity mask from MoGe.
            points3D_world: Nx3 array of sparse 3D points from COLMAP.
            image: The pycolmap.Image object for the current view.
            camera: The pycolmap.Camera object (must be rescaled to match depth map).

        Returns:
            A dictionary containing the 'scaled_depth' and the calculated 'scale'.
        """
        h, w = depth_map.shape
        
        # --- 1. Project sparse points using explicit matrix math ---
        R = image.cam_from_world().rotation.matrix()
        t = image.cam_from_world().translation
        K = camera.calibration_matrix()
        
        # Transform points from world to camera coordinates: P_cam = R * P_world + t
        points3D_camera = (R @ points3D_world.T).T + t
        
        true_depths = points3D_camera[:, 2]
        
        # Filter points behind the camera
        in_front_mask = true_depths > 1e-6
        points3D_camera = points3D_camera[in_front_mask]
        true_depths = true_depths[in_front_mask]

        # Project points from camera to image plane: p_img = K * (P_cam / Z)
        # Normalize by depth
        points2D_normalized = points3D_camera[:, :2] / points3D_camera[:, 2, np.newaxis]
        
        # Apply intrinsics
        u = points2D_normalized[:, 0] * K[0, 0] + K[0, 2]
        v = points2D_normalized[:, 1] * K[1, 1] + K[1, 2]
        projected_kps = np.stack([u, v], axis=-1)
        
        # --- 2. Sample MoGe depth and mask at these 2D locations ---
        kps_x = np.round(projected_kps[:, 0]).astype(int)
        kps_y = np.round(projected_kps[:, 1]).astype(int)

        in_bounds_mask = (kps_x >= 0) & (kps_x < w) & (kps_y >= 0) & (kps_y < h)
        
        kps_x, kps_y = kps_x[in_bounds_mask], kps_y[in_bounds_mask]
        true_depths = true_depths[in_bounds_mask]
        
        moge_depths_at_kps = depth_map[kps_y, kps_x]
        moge_mask_at_kps = mask[kps_y, kps_x]

        # --- 3. Create final filtered set of correspondences ---
        final_valid_mask = (moge_depths_at_kps > 1e-6) & moge_mask_at_kps.astype(bool)
        
        final_true_depths = true_depths[final_valid_mask]
        final_moge_depths = moge_depths_at_kps[final_valid_mask]
        
        num_points = len(final_true_depths)
        if num_points < 10:
            if self.verbose > 0:
                print("Warning: Not enough valid correspondences to find scale. Defaulting to 1.0.")
            return {"scaled_depth": depth_map, "scale": 1.0}

        # --- 4. Transfer data to GPU and solve for the optimal scale ---
        final_true_depths_gpu = cp.asarray(final_true_depths)
        final_moge_depths_gpu = cp.asarray(final_moge_depths)

        b = cp.log(final_true_depths_gpu) - cp.log(final_moge_depths_gpu)
        A = cp.ones((num_points, 1), dtype=cp.float64)
        
        log_scale, _, _, _ = cp.linalg.lstsq(A, b, rcond=None)
        
        if CUPY_AVAILABLE:
            optimized_scale = float(cp.asnumpy(cp.exp(log_scale[0])))
        else:
            optimized_scale = float(np.exp(log_scale[0]))

        if self.verbose > 0:
            print(f"Computed optimal scale: {optimized_scale:.4f} using {num_points} points.")

        return {
            "refined_depth": depth_map * optimized_scale,
            "scale": optimized_scale,
        }