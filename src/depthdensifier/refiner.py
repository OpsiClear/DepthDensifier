"""
Standalone depth map refinement module for aligning depth maps to COLMAP point clouds.
Based on MP-SfM's depth integration and alignment approach.

This module refines monocular depth maps by:
1. Aligning them to sparse 3D points from COLMAP
2. Using surface normals for geometric consistency
3. Optimizing in log-depth space for numerical stability

Usage Examples:

1. Command Line Interface (using tyro):
   python -m depthdensifier.refiner --depth depth.npy --colmap colmap_model/ --image-id 1 --output refined_depth.npy

2. From Python script - Basic usage:
   ```python
   import numpy as np
   from depthdensifier.refiner import DepthRefiner, load_colmap_model

   # Load your depth map and normal map
   depth_map = np.load('depth.npy')
   normal_map = np.load('normals.npy')  # Optional but recommended

   # Load COLMAP reconstruction
   reconstruction = load_colmap_model('colmap_model/')
   image = reconstruction.images[1]  # Image ID
   camera = reconstruction.cameras[image.camera_id]

   # Create refiner and refine
   refiner = DepthRefiner(lambda1=1.0, lambda2=10.0, verbose=1)
   results = refiner.refine_depth(
       depth_map=depth_map,
       normal_map=normal_map,
       points3D=np.array([reconstruction.points3D[pid].xyz
                         for pid in image.point3D_ids if pid != -1]),
       cam_from_world=image.cam_from_world.matrix(),
       K=camera.calibration_matrix()
   )

   refined_depth = results['refined_depth']
   ```

3. From Python script - Using convenience function:
   ```python
   from depthdensifier.refiner import refine_depth_from_colmap, load_colmap_model

   # Load data
   depth_map = np.load('depth.npy')
   normal_map = np.load('normals.npy')
   reconstruction = load_colmap_model('colmap_model/')

   # Refine using COLMAP data structures directly
   results = refine_depth_from_colmap(
       depth_map=depth_map,
       normal_map=normal_map,
       colmap_image=reconstruction.images[1],
       colmap_camera=reconstruction.cameras[reconstruction.images[1].camera_id],
       colmap_points3D=reconstruction.points3D,
       lambda1=1.0,
       lambda2=10.0,
       verbose=1
   )
   ```

4. From Python script - Using configuration dataclass:
   ```python
   from depthdensifier.refiner import DepthRefiner, DepthRefinerConfig

   # Create configuration
   config = DepthRefinerConfig(
       lambda1=1.0,
       lambda2=10.0,
       max_iter=50,
       verbose=1
   )

   # Create refiner with config
   refiner = DepthRefiner(config=config)
   results = refiner.refine_depth(...)
   ```

5. Batch processing multiple images:
   ```python
   from pathlib import Path
   from depthdensifier.refiner import refine_depth_from_colmap, load_colmap_model

   reconstruction = load_colmap_model('colmap_model/')
   depth_dir = Path('depth_maps/')
   output_dir = Path('refined_depths/')

   for image_id, image in reconstruction.images.items():
       depth_path = depth_dir / f'{image.name}_depth.npy'
       if depth_path.exists():
           depth_map = np.load(depth_path)
           results = refine_depth_from_colmap(
               depth_map, None, image,
               reconstruction.cameras[image.camera_id],
               reconstruction.points3D
           )
           np.save(output_dir / f'{image.name}_refined.npy', results['refined_depth'])
   ```

Key Parameters:
- lambda1: Controls depth prior regularization (higher = smoother)
- lambda2: Controls sparse point constraint strength (higher = closer to COLMAP points)
- k_sigmoid: Edge-preserving weight steepness (higher = sharper edges)
- max_iter: Maximum optimization iterations
- verbose: 0=silent, 1=progress, 2=detailed

Returns:
- refined_depth: The refined depth map
- scale: Computed scale factor between input and COLMAP depths
- energy_history: Optimization energy per iteration
- num_iterations: Number of iterations performed
- used_normals: Whether normal constraints were used
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap
import tyro


def setup_sparse_library(use_gpu: bool = True) -> tuple[Any, Any, Any, Any, Any, str]:
    """Set up sparse matrix library (CPU or GPU)."""
    if use_gpu:
        try:
            import cupy as cp
            from cupyx.scipy.sparse import csr_matrix, diags, identity
            from cupyx.scipy.sparse.linalg import cg

            pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
            cp.cuda.set_allocator(pool.malloc)
            print("Using GPU acceleration (CuPy)")
            return cp, csr_matrix, cg, identity, diags, "cuda"
        except ImportError:
            print("CuPy not available, falling back to CPU")

    # CPU fallback
    import numpy as np
    from scipy.sparse import csr_matrix, diags, identity
    from scipy.sparse.linalg import cg

    # Make numpy compatible with cupy interface
    np.asnumpy = np.array
    print("Using CPU (NumPy/SciPy)")
    return np, csr_matrix, cg, identity, diags, "cpu"


# Initialize compute library
cp, csr_matrix, cg, identity, diags, device = setup_sparse_library(use_gpu=True)


@dataclass
class DepthRefinerConfig:
    """
    Configuration for DepthRefiner parameters.

    Examples:
        Default configuration:
        ```python
        config = DepthRefinerConfig()  # Uses default values
        refiner = DepthRefiner(config=config)
        ```

        Custom configuration for high-quality refinement:
        ```python
        config = DepthRefinerConfig(
            lambda1=2.0,           # Stronger depth prior
            lambda2=15.0,          # Stronger sparse constraints
            max_iter=100,          # More iterations
            convergence_tol=1e-4,  # Tighter convergence
            verbose=2              # Detailed output
        )
        ```

        Configuration for fast refinement:
        ```python
        config = DepthRefinerConfig(
            max_iter=20,           # Fewer iterations
            cg_max_iter=500,       # Faster CG solver
            convergence_tol=1e-2,  # Looser convergence
            verbose=0              # Silent
        )
        ```
    """

    lambda1: float = 1.0  # Prior regularization weight
    lambda2: float = 10.0  # Sparse point constraint weight
    k_sigmoid: float = 10.0  # Sigmoid steepness for discontinuity
    max_iter: int = 50
    cg_max_iter: int = 1000
    cg_tol: float = 1e-5
    convergence_tol: float = 1e-3
    scale_filter_factor: float = 2.0
    verbose: int = 1


@dataclass
class MainConfig:
    """
    Configuration for main CLI function.
    
    Examples:
        Command line usage with tyro:
        ```bash
        # Basic usage
        python -m depthdensifier.refiner \
            --depth depth.npy \
            --colmap colmap_model/ \
            --image-id 1 \
            --output refined_depth.npy
        
        # With normals and custom parameters
        python -m depthdensifier.refiner \
            --depth depth.npy \
            --normals normals.npy \
            --colmap colmap_model/ \
            --image-id 1 \
            --output refined_depth.npy \
            --lambda1 2.0 \
            --lambda2 5.0 \
            --verbose 2
        ```
        
        Programmatic usage:
        ```python
        config = MainConfig(
            depth='depth.npy',
            colmap='colmap_model/',
            image_id=1,
            output='refined_depth.npy',
            normals='normals.npy',
            lambda1=1.5,
            lambda2=8.0,
            verbose=1
        )
        ```
    """

    depth: str  # Path to depth map (NPY or image)
    colmap: str  # Path to COLMAP model
    image_id: int  # COLMAP image ID
    output: str  # Output path for refined depth
    normals: str | None = None  # Path to normal map (NPY) - optional but recommended
    lambda1: float = 1.0  # Prior weight
    lambda2: float = 10.0  # Sparse constraint weight
    verbose: int = 1  # Verbosity level


class DepthRefiner:
    """
    Refines depth maps by aligning them to COLMAP point clouds using normal constraints.

    This class implements a depth refinement algorithm that aligns monocular depth estimates
    with sparse 3D points from COLMAP reconstructions. It uses surface normals for geometric
    consistency and optimizes in log-depth space for numerical stability.

    Examples:
        Basic usage with default parameters:
        ```python
        refiner = DepthRefiner()
        results = refiner.refine_depth(depth_map, normal_map, points3D, cam_from_world, K)
        refined_depth = results['refined_depth']
        ```

        Custom configuration:
        ```python
        refiner = DepthRefiner(
            lambda1=2.0,      # Stronger depth prior
            lambda2=5.0,      # Weaker sparse constraints
            max_iter=100,     # More iterations
            verbose=2         # Detailed output
        )
        ```

        Using configuration dataclass:
        ```python
        config = DepthRefinerConfig(lambda1=1.5, lambda2=8.0, verbose=1)
        refiner = DepthRefiner(config=config)
        ```

        With uncertainty maps:
        ```python
        results = refiner.refine_depth(
            depth_map=depth,
            normal_map=normals,
            points3D=points,
            cam_from_world=pose,
            K=intrinsics,
            depth_uncertainty=depth_std,      # Higher values = less trust
            normal_uncertainty=normal_std     # Per-component uncertainty
        )
        ```
    """

    def __init__(
        self,
        config: DepthRefinerConfig | None = None,
        lambda1: float | None = None,  # Prior regularization weight
        lambda2: float | None = None,  # Sparse point constraint weight
        k_sigmoid: float | None = None,  # Sigmoid steepness for discontinuity
        max_iter: int | None = None,
        cg_max_iter: int | None = None,
        cg_tol: float | None = None,
        convergence_tol: float | None = None,
        scale_filter_factor: float | None = None,
        verbose: int | None = None,
    ):
        """
        Initialize depth refiner.

        :param config: Configuration object with all parameters
        :param lambda1: Weight for depth prior regularization
        :param lambda2: Weight for sparse 3D point constraints
        :param k_sigmoid: Steepness of sigmoid for edge-preserving weights
        :param max_iter: Maximum optimization iterations
        :param cg_max_iter: Maximum conjugate gradient iterations
        :param cg_tol: Conjugate gradient tolerance
        :param convergence_tol: Convergence tolerance for energy
        :param scale_filter_factor: Factor for filtering outlier scales
        :param verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        if config is None:
            config = DepthRefinerConfig()

        # Use provided parameters or fall back to config defaults
        self.lambda1 = lambda1 if lambda1 is not None else config.lambda1
        self.lambda2 = lambda2 if lambda2 is not None else config.lambda2
        self.k_sigmoid = k_sigmoid if k_sigmoid is not None else config.k_sigmoid
        self.max_iter = max_iter if max_iter is not None else config.max_iter
        self.cg_max_iter = (
            cg_max_iter if cg_max_iter is not None else config.cg_max_iter
        )
        self.cg_tol = cg_tol if cg_tol is not None else config.cg_tol
        self.convergence_tol = (
            convergence_tol if convergence_tol is not None else config.convergence_tol
        )
        self.scale_filter_factor = (
            scale_filter_factor
            if scale_filter_factor is not None
            else config.scale_filter_factor
        )
        self.verbose = verbose if verbose is not None else config.verbose

    def project_points_to_image(
        self, points3D: np.ndarray, cam_from_world: np.ndarray, K: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project 3D points to image plane.

        :param points3D: Nx3 array of 3D points in world coordinates
        :param cam_from_world: 3x4 camera extrinsics matrix
        :param K: 3x3 camera intrinsics matrix
        :return: tuple of (pts2d, depths, valid) where pts2d is Nx2 array of 2D image points,
                depths is N array of depths in camera frame, valid is N array of boolean validity mask
        """
        # Convert to homogeneous coordinates
        points3D_h = np.hstack([points3D, np.ones((points3D.shape[0], 1))])

        # Transform to camera coordinates
        H = np.vstack([cam_from_world, [0, 0, 0, 1]])
        points_cam = (H @ points3D_h.T)[:3, :].T

        # Get depths
        depths = points_cam[:, 2].copy()

        # Project to image
        valid = depths > 1e-6
        pts2d = np.zeros((len(depths), 2))
        pts2d[valid] = ((K @ (points_cam[valid] / depths[valid, None]).T).T)[:, :2]

        return pts2d, depths, valid

    def sample_depth_at_points(
        self, depth_map: np.ndarray, pts2d: np.ndarray
    ) -> np.ndarray:
        """
        Sample depth values at 2D points using bilinear interpolation.

        :param depth_map: HxW depth map
        :param pts2d: Nx2 array of 2D points
        :return: N array of sampled depth values
        """
        h, w = depth_map.shape
        x = pts2d[:, 0]
        y = pts2d[:, 1]

        # Bilinear interpolation
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        # Clip to image bounds
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
        y1 = np.clip(y1, 0, h - 1)

        # Interpolation weights
        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)

        # Sample depths
        depths = (
            wa * depth_map[y0, x0]
            + wb * depth_map[y0, x1]
            + wc * depth_map[y1, x0]
            + wd * depth_map[y1, x1]
        )

        return depths

    def compute_gradient_operators(
        self, shape: tuple[int, int]
    ) -> tuple[Any, Any, Any, Any]:
        """
        Create sparse matrices for computing gradients.

        :param shape: (H, W) shape of depth map
        :return: tuple of (Dx_pos, Dx_neg, Dy_pos, Dy_neg) gradient operators
        """
        h, w = shape
        n = h * w

        # Create masks for valid neighbors

        # Horizontal gradients (x-direction)
        has_right = cp.zeros((h, w), dtype=bool)
        has_right[:, :-1] = True
        has_left = cp.zeros((h, w), dtype=bool)
        has_left[:, 1:] = True

        # Vertical gradients (y-direction)
        has_bottom = cp.zeros((h, w), dtype=bool)
        has_bottom[:-1, :] = True
        has_top = cp.zeros((h, w), dtype=bool)
        has_top[1:, :] = True

        # Flatten masks
        has_right_flat = has_right.flatten()
        has_left_flat = has_left.flatten()
        has_bottom_flat = has_bottom.flatten()
        has_top_flat = has_top.flatten()

        # Create index arrays
        idx = cp.arange(n).reshape((h, w))

        # Dx positive (right difference)
        idx_right = cp.zeros((h, w), dtype=int)
        idx_right[:, :-1] = idx[:, 1:]
        data = cp.ones(has_right_flat.sum().item())
        indices = cp.concatenate(
            [idx[has_right].flatten(), idx_right[has_right].flatten()]
        )
        indptr = cp.concatenate(
            [cp.array([0]), cp.cumsum(has_right_flat.astype(int) * 2)]
        )
        Dx_pos = csr_matrix(
            (cp.concatenate([-data, data]), indices, indptr), shape=(n, n)
        )

        # Dx negative (left difference)
        idx_left = cp.zeros((h, w), dtype=int)
        idx_left[:, 1:] = idx[:, :-1]
        data = cp.ones(has_left_flat.sum().item())
        indices = cp.concatenate(
            [idx_left[has_left].flatten(), idx[has_left].flatten()]
        )
        indptr = cp.concatenate(
            [cp.array([0]), cp.cumsum(has_left_flat.astype(int) * 2)]
        )
        Dx_neg = csr_matrix(
            (cp.concatenate([-data, data]), indices, indptr), shape=(n, n)
        )

        # Dy positive (bottom difference)
        idx_bottom = cp.zeros((h, w), dtype=int)
        idx_bottom[:-1, :] = idx[1:, :]
        data = cp.ones(has_bottom_flat.sum().item())
        indices = cp.concatenate(
            [idx[has_bottom].flatten(), idx_bottom[has_bottom].flatten()]
        )
        indptr = cp.concatenate(
            [cp.array([0]), cp.cumsum(has_bottom_flat.astype(int) * 2)]
        )
        Dy_pos = csr_matrix(
            (cp.concatenate([-data, data]), indices, indptr), shape=(n, n)
        )

        # Dy negative (top difference)
        idx_top = cp.zeros((h, w), dtype=int)
        idx_top[1:, :] = idx[:-1, :]
        data = cp.ones(has_top_flat.sum().item())
        indices = cp.concatenate([idx_top[has_top].flatten(), idx[has_top].flatten()])
        indptr = cp.concatenate(
            [cp.array([0]), cp.cumsum(has_top_flat.astype(int) * 2)]
        )
        Dy_neg = csr_matrix(
            (cp.concatenate([-data, data]), indices, indptr), shape=(n, n)
        )

        return Dx_pos, Dx_neg, Dy_pos, Dy_neg

    def sigmoid(self, x: np.ndarray, k: float = 1.0) -> np.ndarray:
        """Sigmoid function for edge-preserving weights."""
        x_clipped = cp.clip(-k * x, -709, 709)
        return 1 / (1 + cp.exp(x_clipped))

    def compute_edge_weights(
        self, z: Any, Dx_pos: Any, Dx_neg: Any, Dy_pos: Any, Dy_neg: Any
    ) -> tuple[Any, Any]:
        """
        Compute edge-preserving weights based on depth gradients.

        :param z: Log-depth values
        :param Dx_pos: Positive x-direction gradient operator
        :param Dx_neg: Negative x-direction gradient operator
        :param Dy_pos: Positive y-direction gradient operator
        :param Dy_neg: Negative y-direction gradient operator
        :return: tuple of (wx, wy) edge weights for x and y directions
        """
        # Replace any NaN or inf values in z with 0
        z_clean = cp.where(cp.isfinite(z), z, 0)
        
        # Compute gradients
        grad_x_pos = Dx_pos.dot(z_clean)
        grad_x_neg = Dx_neg.dot(z_clean)
        grad_y_pos = Dy_pos.dot(z_clean)
        grad_y_neg = Dy_neg.dot(z_clean)

        # Compute differences safely
        diff_x = grad_x_neg**2 - grad_x_pos**2
        diff_y = grad_y_neg**2 - grad_y_pos**2
        
        # Replace NaN/inf with 0 before sigmoid
        diff_x = cp.where(cp.isfinite(diff_x), diff_x, 0)
        diff_y = cp.where(cp.isfinite(diff_y), diff_y, 0)
        
        # Compute weights using sigmoid
        wx = self.sigmoid(diff_x, self.k_sigmoid)
        wy = self.sigmoid(diff_y, self.k_sigmoid)
        
        # Ensure weights are valid (between 0 and 1)
        wx = cp.clip(wx, 0.0, 1.0)
        wy = cp.clip(wy, 0.0, 1.0)

        return wx, wy

    def refine_depth(
        self,
        depth_map: np.ndarray,
        normal_map: np.ndarray | None,
        points3D: np.ndarray,
        cam_from_world: np.ndarray,
        K: np.ndarray,
        mask: np.ndarray | None = None,
        depth_uncertainty: np.ndarray | None = None,
        normal_uncertainty: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Refine depth map to align with COLMAP point cloud.

        :param depth_map: HxW depth map to refine (in meters)
        :param normal_map: HxWx3 surface normal map (optional, but highly recommended)
                          Normals should be in camera coordinates pointing towards camera
                          i.e., n_z should be negative for surfaces facing the camera
        :param points3D: Nx3 array of 3D points in world coordinates
        :param cam_from_world: 3x4 camera extrinsics matrix
        :param K: 3x3 camera intrinsics matrix
        :param mask: HxW mask of valid pixels (optional)
        :param depth_uncertainty: HxW depth uncertainty map (optional)
        :param normal_uncertainty: HxWx3 normal uncertainty map (optional)
        :return: Dictionary with refined_depth (HxW refined depth map), scale (computed scale factor),
                energy_history (optimization energy history), num_iterations (number of iterations performed),
                and used_normals (whether normal constraints were used)

        Examples:
            Basic refinement with normals:
            ```python
            refiner = DepthRefiner(lambda1=1.0, lambda2=10.0)

            # Prepare data
            depth_map = np.load('monocular_depth.npy')  # Shape: (H, W)
            normal_map = np.load('surface_normals.npy')  # Shape: (H, W, 3)
            points3D = colmap_points  # Shape: (N, 3) - world coordinates
            cam_from_world = image.cam_from_world.matrix()  # Shape: (3, 4)
            K = camera.calibration_matrix()  # Shape: (3, 3)

            # Refine
            results = refiner.refine_depth(depth_map, normal_map, points3D, cam_from_world, K)

            # Access results
            refined_depth = results['refined_depth']  # Same shape as input
            scale_factor = results['scale']           # Depth scale correction
            convergence = results['energy_history']   # Energy per iteration
            ```

            Refinement without normals (smoothness only):
            ```python
            results = refiner.refine_depth(
                depth_map=depth_map,
                normal_map=None,      # Will use smoothness regularization
                points3D=points3D,
                cam_from_world=cam_from_world,
                K=K
            )
            ```

            With uncertainty weighting:
            ```python
            # Higher uncertainty = lower weight in optimization
            depth_uncertainty = np.ones_like(depth_map) * 0.1  # 10cm std
            normal_uncertainty = np.ones_like(normal_map) * 0.05  # 5° std per component

            results = refiner.refine_depth(
                depth_map, normal_map, points3D, cam_from_world, K,
                depth_uncertainty=depth_uncertainty,
                normal_uncertainty=normal_uncertainty
            )
            ```
        """
        h, w = depth_map.shape
        fx, fy = K[0, 0], K[1, 1]

        use_normals = normal_map is not None

        # Project 3D points to image
        if self.verbose > 0:
            print(f"Projecting {len(points3D)} 3D points to image...")
        pts2d, depths3d, valid = self.project_points_to_image(
            points3D, cam_from_world, K
        )

        # Filter points within image bounds
        in_bounds = (
            (pts2d[:, 0] >= 0)
            & (pts2d[:, 0] < w)
            & (pts2d[:, 1] >= 0)
            & (pts2d[:, 1] < h)
            & valid
        )
        pts2d = pts2d[in_bounds]
        depths3d = depths3d[in_bounds]

        if len(pts2d) == 0:
            print("Warning: No 3D points project into image bounds")
            return {
                "refined_depth": depth_map,
                "scale": 1.0,
                "energy_history": [],
                "num_iterations": 0,
            }

        # Sample depth at projected points
        depth_at_pts = self.sample_depth_at_points(depth_map, pts2d)
        if mask is not None:
            mask_at_pts = self.sample_depth_at_points(mask.astype(float), pts2d) > 0.5

        # Combine all validity checks
        valid_scale = (depth_at_pts > 1e-3) & (depths3d > 1e-3) & (mask_at_pts if mask is not None else True)

        if valid_scale.sum() == 0:
            print("Warning: No valid depth correspondences")
            scale = 1.0
        else:
            scale = float(np.median(depths3d[valid_scale] / depth_at_pts[valid_scale]))

        if self.verbose > 0:
            print(f"Computed scale factor: {scale:.4f}")
            print(f"Using {valid_scale.sum()} valid correspondences")

        # Move to GPU if available
        depth_map_gpu = cp.asarray(depth_map * scale, dtype=np.float64)
        
        # Clean up any NaN or inf values
        depth_map_gpu = cp.where(cp.isfinite(depth_map_gpu), depth_map_gpu, 1.0)
        depth_map_gpu = cp.maximum(depth_map_gpu, 1e-6)  # Ensure positive

        # Initialize in log space
        z = cp.log(depth_map_gpu + 1e-6).flatten()
        z_prior = z.copy()
        
        # Clean up any remaining NaN/inf
        z = cp.where(cp.isfinite(z), z, 0.0)
        z_prior = cp.where(cp.isfinite(z_prior), z_prior, 0.0)

        # Process normals if provided
        if use_normals:
            normal_map_gpu = cp.asarray(normal_map, dtype=np.float64)

            # Normalize normals if needed
            norm = cp.linalg.norm(normal_map_gpu, axis=2, keepdims=True)
            normal_map_gpu = normal_map_gpu / (norm + 1e-8)

            # Extract normal components
            nx = normal_map_gpu[..., 0].flatten()
            ny = normal_map_gpu[..., 1].flatten()
            nz = normal_map_gpu[..., 2].flatten()

            # If normals point away from camera, flip them
            if cp.median(nz) > 0:
                nz = -nz
                if self.verbose > 0:
                    print("Flipped normals to point towards camera")

            # Create coordinate grids (correct order for numpy indexing)
            yy, xx = cp.meshgrid(cp.arange(h), cp.arange(w), indexing="ij")
            xx = xx.flatten()
            yy = yy.flatten()

            # Compute normal-based gradients (depth gradients from normals)
            # Based on perspective projection: z_x = -nx/nz * z/fx, z_y = -ny/nz * z/fy
            # In log space: d(log z)/dx = -nx/(nz*fx), d(log z)/dy = -ny/(nz*fy)
            nz_safe = cp.clip(cp.abs(nz), 0.1, 1.0)  # Avoid division by small values
            target_grad_x = -nx / (nz_safe * fx)
            target_grad_y = -ny / (nz_safe * fy)
        else:
            # Without normals, use simple smoothness
            target_grad_x = cp.zeros_like(z)
            target_grad_y = cp.zeros_like(z)

        # Setup uncertainties
        if depth_uncertainty is None:
            depth_precision = cp.ones_like(z)
        else:
            depth_precision = cp.asarray(1.0 / (depth_uncertainty.flatten() + 1e-6))
            depth_precision *= depth_map_gpu.flatten() ** 2  # For log-space

        if use_normals:
            if normal_uncertainty is None:
                normal_precision_x = cp.ones_like(z)
                normal_precision_y = cp.ones_like(z)
            else:
                normal_precision_x = cp.asarray(
                    1.0 / (normal_uncertainty[..., 0].flatten() + 1e-6)
                )
                normal_precision_y = cp.asarray(
                    1.0 / (normal_uncertainty[..., 1].flatten() + 1e-6)
                )
        else:
            # Without normals, use uniform smoothness weight
            normal_precision_x = (
                cp.ones_like(z) * 0.1
            )  # Lower weight for simple smoothness
            normal_precision_y = cp.ones_like(z) * 0.1

        # Prepare sparse constraints
        sparse_ids = np.ravel_multi_index(
            (
                np.clip(pts2d[:, 1].astype(int), 0, h - 1),
                np.clip(pts2d[:, 0].astype(int), 0, w - 1),
            ),
            (h, w),
        )
        sparse_ids = cp.asarray(sparse_ids)
        sparse_depth = cp.log(cp.asarray(depths3d) + 1e-6)
        sparse_precision = cp.ones(len(sparse_ids))

        # Filter outliers based on scale consistency
        if self.scale_filter_factor > 1.0:
            div = cp.exp(sparse_depth) / cp.exp(z_prior[sparse_ids])
            valid_sparse = (div < self.scale_filter_factor) & (
                div > 1 / self.scale_filter_factor
            )
            if device == "cuda":
                valid_sparse = valid_sparse.get()
            sparse_ids = sparse_ids[valid_sparse]
            sparse_depth = sparse_depth[valid_sparse]
            sparse_precision = sparse_precision[valid_sparse]
            if self.verbose > 0:
                print(f"Filtered to {len(sparse_ids)} inlier correspondences")

        # Get gradient operators
        Dx_pos, Dx_neg, Dy_pos, Dy_neg = self.compute_gradient_operators((h, w))

        # Optimization loop
        energy_history = []
        for iteration in range(self.max_iter):
            # Compute edge weights
            wx, wy = self.compute_edge_weights(z, Dx_pos, Dx_neg, Dy_pos, Dy_neg)

            # Weight normals by edge weights and precision
            wx_pos = wx * normal_precision_x
            wx_neg = (1 - wx) * normal_precision_x
            wy_pos = wy * normal_precision_y
            wy_neg = (1 - wy) * normal_precision_y

            # Build system matrix A
            A = self.lambda1 * diags(depth_precision)

            # Add gradient terms
            A += Dx_pos.T @ diags(wx_pos) @ Dx_pos
            A += Dx_neg.T @ diags(wx_neg) @ Dx_neg
            A += Dy_pos.T @ diags(wy_pos) @ Dy_pos
            A += Dy_neg.T @ diags(wy_neg) @ Dy_neg

            # Add sparse constraints
            if len(sparse_ids) > 0:
                sparse_mat = csr_matrix(
                    (sparse_precision, (sparse_ids, sparse_ids)), shape=(len(z), len(z))
                )
                A += self.lambda2 * sparse_mat

            # Build right-hand side b
            b = self.lambda1 * depth_precision * z_prior
            b += Dx_pos.T @ (wx_pos * target_grad_x)
            b += Dx_neg.T @ (wx_neg * target_grad_x)
            b += Dy_pos.T @ (wy_pos * target_grad_y)
            b += Dy_neg.T @ (wy_neg * target_grad_y)

            if len(sparse_ids) > 0:
                b[sparse_ids] += self.lambda2 * sparse_precision * sparse_depth

            # Solve using conjugate gradient
            if device == "cuda":
                z_new, info = cg(A, b, x0=z, maxiter=self.cg_max_iter, tol=self.cg_tol)
            else:
                # scipy uses rtol instead of tol
                z_new, info = cg(A, b, x0=z, maxiter=self.cg_max_iter, rtol=self.cg_tol)

            # Compute energy
            energy = float(cp.sum((z_new - z_prior) ** 2))
            energy_history.append(energy)

            # Check convergence
            if iteration > 0:
                relative_change = abs(energy - energy_history[-2]) / (
                    energy_history[-2] + 1e-10
                )
                if relative_change < self.convergence_tol:
                    if self.verbose > 0:
                        print(f"Converged at iteration {iteration + 1}")
                    break

            z = z_new

            if self.verbose > 1 and iteration % 10 == 0:
                print(
                    f"Iteration {iteration + 1}/{self.max_iter}, Energy: {energy:.6f}"
                )

        # Convert back from log space
        refined_depth = cp.exp(z.reshape((h, w)))

        # Move back to CPU if needed
        if device == "cuda":
            refined_depth = refined_depth.get()

        return {
            "refined_depth": np.asarray(refined_depth),
            "scale": scale,
            "energy_history": energy_history,
            "num_iterations": iteration + 1,
            "used_normals": use_normals,
        }


def load_colmap_model(model_path: str | Path) -> pycolmap.Reconstruction:
    """
    Load COLMAP reconstruction from path.

    :param model_path: Path to COLMAP model directory or binary file
    :return: COLMAP reconstruction object

    Examples:
        Load from directory (text format):
        ```python
        reconstruction = load_colmap_model('colmap_model/')
        print(f"Loaded {len(reconstruction.images)} images")
        print(f"Loaded {len(reconstruction.points3D)} 3D points")
        ```

        Load from binary file:
        ```python
        reconstruction = load_colmap_model('model.bin')

        # Access specific image
        image_id = 1
        if image_id in reconstruction.images:
            image = reconstruction.images[image_id]
            camera = reconstruction.cameras[image.camera_id]
            print(f"Image: {image.name}, Camera model: {camera.model_name}")
        ```

        Iterate through all images:
        ```python
        reconstruction = load_colmap_model('colmap_model/')
        for image_id, image in reconstruction.images.items():
            camera = reconstruction.cameras[image.camera_id]
            num_points = sum(1 for p in image.points2D if p.has_point3D())
            print(f"Image {image.name}: {num_points} 3D points visible")
        ```
    """
    if Path(model_path).is_dir():
        return pycolmap.Reconstruction(model_path)
    else:
        reconstruction = pycolmap.Reconstruction()
        reconstruction.read_binary(model_path)
        return reconstruction


def refine_depth_from_colmap(
    depth_map: np.ndarray,
    normal_map: np.ndarray | None,
    colmap_image: pycolmap.Image,
    colmap_camera: pycolmap.Camera,
    colmap_points3D: dict[int, Any],
    **kwargs,
) -> dict[str, Any]:
    """
    Convenience function to refine depth using COLMAP data structures.

    :param depth_map: HxW depth map
    :param normal_map: HxWx3 normal map (optional but recommended)
    :param colmap_image: COLMAP image object
    :param colmap_camera: COLMAP camera object
    :param colmap_points3D: Dictionary of COLMAP 3D points
    :param kwargs: Additional arguments for DepthRefiner
    :return: Refinement results dictionary

    Examples:
        Basic usage with COLMAP data:
        ```python
        import numpy as np
        from depthdensifier.refiner import refine_depth_from_colmap, load_colmap_model

        # Load COLMAP reconstruction
        reconstruction = load_colmap_model('path/to/colmap/model/')

        # Get specific image and camera
        image_id = 1
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]

        # Load depth and normal maps
        depth_map = np.load(f'depth_{image.name}.npy')
        normal_map = np.load(f'normals_{image.name}.npy')  # Optional

        # Refine depth
        results = refine_depth_from_colmap(
            depth_map=depth_map,
            normal_map=normal_map,
            colmap_image=image,
            colmap_camera=camera,
            colmap_points3D=reconstruction.points3D,
            lambda1=1.0,      # Depth prior weight
            lambda2=10.0,     # Sparse constraint weight
            verbose=1
        )

        # Save result
        np.save(f'refined_{image.name}.npy', results['refined_depth'])
        ```

        Batch processing all images in reconstruction:
        ```python
        for image_id, image in reconstruction.images.items():
            # Skip if no depth map available
            depth_path = f'depths/depth_{image.name}.npy'
            if not os.path.exists(depth_path):
                continue

            depth_map = np.load(depth_path)
            camera = reconstruction.cameras[image.camera_id]

            results = refine_depth_from_colmap(
                depth_map, None, image, camera, reconstruction.points3D,
                lambda1=2.0, lambda2=5.0  # Custom parameters
            )

            print(f"Image {image.name}: scale={results['scale']:.3f}, "
                  f"iterations={results['num_iterations']}")
        ```
    """
    # Extract camera parameters
    K = colmap_camera.calibration_matrix()
    cam_from_world = colmap_image.cam_from_world.matrix()

    # Extract visible 3D points
    point_ids = [p.point3D_id for p in colmap_image.points2D if p.has_point3D()]
    points3D = np.array(
        [colmap_points3D[pid].xyz for pid in point_ids if pid in colmap_points3D]
    )

    # Create refiner and refine
    refiner = DepthRefiner(**kwargs)
    return refiner.refine_depth(depth_map, normal_map, points3D, cam_from_world, K)


def main() -> None:
    """Example usage of depth refiner."""
    config = tyro.cli(MainConfig)

    # Load data
    print("Loading depth map...")
    if config.depth.endswith(".npy"):
        depth_map = np.load(config.depth)
    else:
        depth_map = (
            cv2.imread(config.depth, cv2.IMREAD_ANYDEPTH) / 1000.0
        )  # Assume mm to m

    # Load normals if provided
    if config.normals:
        print("Loading normal map...")
        normal_map = np.load(config.normals)
    else:
        print("No normal map provided - using smoothness regularization only")
        normal_map = None

    print("Loading COLMAP model...")
    reconstruction = load_colmap_model(config.colmap)

    if config.image_id not in reconstruction.images:
        raise ValueError(f"Image ID {config.image_id} not found in reconstruction")

    image = reconstruction.images[config.image_id]
    camera = reconstruction.cameras[image.camera_id]

    # Refine depth
    print("Refining depth map...")
    results = refine_depth_from_colmap(
        depth_map,
        normal_map,
        image,
        camera,
        reconstruction.points3D,
        lambda1=config.lambda1,
        lambda2=config.lambda2,
        verbose=config.verbose,
    )

    # Save results
    print(f"Saving refined depth to {config.output}")
    np.save(config.output, results["refined_depth"])

    print("Refinement complete!")
    print(f"  Scale factor: {results['scale']:.4f}")
    print(f"  Iterations: {results['num_iterations']}")
    print(f"  Used normals: {results['used_normals']}")
    if results["energy_history"]:
        print(f"  Final energy: {results['energy_history'][-1]:.6f}")


if __name__ == "__main__":
    main()
