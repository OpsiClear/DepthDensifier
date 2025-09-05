"""
Standalone depth map refinement module for aligning depth maps to COLMAP point clouds.
Based on MP-SfM's depth integration and alignment approach.

This module refines monocular depth maps by:
1. Aligning them to sparse 3D points from COLMAP
2. Using surface normals for geometric consistency
3. Optimizing in log-depth space for numerical stability
"""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pycolmap


def setup_sparse_library(use_gpu=True):
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


class DepthRefiner:
    """
    Refines depth maps by aligning them to COLMAP point clouds using normal constraints.
    """

    def __init__(
        self,
        lambda1: float = 1.0,  # Prior regularization weight
        lambda2: float = 10.0,  # Sparse point constraint weight
        k_sigmoid: float = 10.0,  # Sigmoid steepness for discontinuity
        max_iter: int = 50,
        cg_max_iter: int = 1000,
        cg_tol: float = 1e-5,
        convergence_tol: float = 1e-3,
        scale_filter_factor: float = 2.0,
        verbose: int = 1,
    ):
        """
        Initialize depth refiner.

        Args:
            lambda1: Weight for depth prior regularization
            lambda2: Weight for sparse 3D point constraints
            k_sigmoid: Steepness of sigmoid for edge-preserving weights
            max_iter: Maximum optimization iterations
            cg_max_iter: Maximum conjugate gradient iterations
            cg_tol: Conjugate gradient tolerance
            convergence_tol: Convergence tolerance for energy
            scale_filter_factor: Factor for filtering outlier scales
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.k_sigmoid = k_sigmoid
        self.max_iter = max_iter
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        self.convergence_tol = convergence_tol
        self.scale_filter_factor = scale_filter_factor
        self.verbose = verbose

    def project_points_to_image(
        self, points3D: np.ndarray, cam_from_world: np.ndarray, K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points to image plane.

        Args:
            points3D: Nx3 array of 3D points in world coordinates
            cam_from_world: 3x4 camera extrinsics matrix
            K: 3x3 camera intrinsics matrix

        Returns:
            pts2d: Nx2 array of 2D image points
            depths: N array of depths in camera frame
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

        Args:
            depth_map: HxW depth map
            pts2d: Nx2 array of 2D points

        Returns:
            depths: N array of sampled depth values
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

    def compute_gradient_operators(self, shape: tuple) -> Tuple:
        """
        Create sparse matrices for computing gradients.

        Args:
            shape: (H, W) shape of depth map

        Returns:
            Dx_pos, Dx_neg, Dy_pos, Dy_neg: Gradient operators
        """
        h, w = shape
        n = h * w

        # Create masks for valid neighbors
        mask = cp.ones((h, w), dtype=bool)

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

    def compute_edge_weights(self, z, Dx_pos, Dx_neg, Dy_pos, Dy_neg):
        """
        Compute edge-preserving weights based on depth gradients.

        Args:
            z: Log-depth values
            Dx_pos, Dx_neg, Dy_pos, Dy_neg: Gradient operators

        Returns:
            wx, wy: Edge weights for x and y directions
        """
        # Compute gradients
        grad_x_pos = Dx_pos.dot(z)
        grad_x_neg = Dx_neg.dot(z)
        grad_y_pos = Dy_pos.dot(z)
        grad_y_neg = Dy_neg.dot(z)

        # Compute weights using sigmoid
        wx = self.sigmoid(grad_x_neg**2 - grad_x_pos**2, self.k_sigmoid)
        wy = self.sigmoid(grad_y_neg**2 - grad_y_pos**2, self.k_sigmoid)

        return wx, wy

    def refine_depth(
        self,
        depth_map: np.ndarray,
        normal_map: Optional[np.ndarray],
        points3D: np.ndarray,
        cam_from_world: np.ndarray,
        K: np.ndarray,
        depth_uncertainty: Optional[np.ndarray] = None,
        normal_uncertainty: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Refine depth map to align with COLMAP point cloud.

        Args:
            depth_map: HxW depth map to refine (in meters)
            normal_map: HxWx3 surface normal map (optional, but highly recommended)
                       Normals should be in camera coordinates pointing towards camera
                       i.e., n_z should be negative for surfaces facing the camera
            points3D: Nx3 array of 3D points in world coordinates
            cam_from_world: 3x4 camera extrinsics matrix
            K: 3x3 camera intrinsics matrix
            depth_uncertainty: HxW depth uncertainty map (optional)
            normal_uncertainty: HxWx3 normal uncertainty map (optional)

        Returns:
            dict with:
                - refined_depth: HxW refined depth map
                - scale: Computed scale factor
                - energy_history: Optimization energy history
                - num_iterations: Number of iterations performed
                - used_normals: Whether normal constraints were used
        """
        h, w = depth_map.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

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

        # Compute robust scale using median
        valid_scale = (depth_at_pts > 1e-3) & (depths3d > 1e-3)
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

        # Initialize in log space
        z = cp.log(depth_map_gpu + 1e-6).flatten()
        z_prior = z.copy()

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
            uu = xx - cx
            vv = yy - cy

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
            z_new, info = cg(A, b, x0=z, maxiter=self.cg_max_iter, tol=self.cg_tol)

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


def load_colmap_model(model_path: str) -> pycolmap.Reconstruction:
    """Load COLMAP reconstruction from path."""
    if Path(model_path).is_dir():
        return pycolmap.Reconstruction(model_path)
    else:
        reconstruction = pycolmap.Reconstruction()
        reconstruction.read_binary(model_path)
        return reconstruction


def refine_depth_from_colmap(
    depth_map: np.ndarray,
    normal_map: Optional[np.ndarray],
    colmap_image: pycolmap.Image,
    colmap_camera: pycolmap.Camera,
    colmap_points3D: dict,
    **kwargs,
) -> Dict:
    """
    Convenience function to refine depth using COLMAP data structures.

    Args:
        depth_map: HxW depth map
        normal_map: HxWx3 normal map (optional but recommended)
        colmap_image: COLMAP image object
        colmap_camera: COLMAP camera object
        colmap_points3D: Dictionary of COLMAP 3D points
        **kwargs: Additional arguments for DepthRefiner

    Returns:
        Refinement results dictionary
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


def main():
    """Example usage of depth refiner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Refine depth maps using COLMAP point clouds"
    )
    parser.add_argument(
        "--depth", required=True, help="Path to depth map (NPY or image)"
    )
    parser.add_argument(
        "--normals", help="Path to normal map (NPY) - optional but recommended"
    )
    parser.add_argument("--colmap", required=True, help="Path to COLMAP model")
    parser.add_argument("--image_id", type=int, required=True, help="COLMAP image ID")
    parser.add_argument("--output", required=True, help="Output path for refined depth")
    parser.add_argument("--lambda1", type=float, default=1.0, help="Prior weight")
    parser.add_argument(
        "--lambda2", type=float, default=10.0, help="Sparse constraint weight"
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")

    args = parser.parse_args()

    # Load data
    print("Loading depth map...")
    if args.depth.endswith(".npy"):
        depth_map = np.load(args.depth)
    else:
        depth_map = (
            cv2.imread(args.depth, cv2.IMREAD_ANYDEPTH) / 1000.0
        )  # Assume mm to m

    # Load normals if provided
    if args.normals:
        print("Loading normal map...")
        normal_map = np.load(args.normals)
    else:
        print("No normal map provided - using smoothness regularization only")
        normal_map = None

    print("Loading COLMAP model...")
    reconstruction = load_colmap_model(args.colmap)

    if args.image_id not in reconstruction.images:
        raise ValueError(f"Image ID {args.image_id} not found in reconstruction")

    image = reconstruction.images[args.image_id]
    camera = reconstruction.cameras[image.camera_id]

    # Refine depth
    print("Refining depth map...")
    results = refine_depth_from_colmap(
        depth_map,
        normal_map,
        image,
        camera,
        reconstruction.points3D,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        verbose=args.verbose,
    )

    # Save results
    print(f"Saving refined depth to {args.output}")
    np.save(args.output, results["refined_depth"])

    print(f"Refinement complete!")
    print(f"  Scale factor: {results['scale']:.4f}")
    print(f"  Iterations: {results['num_iterations']}")
    print(f"  Used normals: {results['used_normals']}")
    if results["energy_history"]:
        print(f"  Final energy: {results['energy_history'][-1]:.6f}")


if __name__ == "__main__":
    main()
