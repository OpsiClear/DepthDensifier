"""
Multi-view consistency filtering for removing floating points from densified point clouds.

This module provides fast vectorized filtering of 3D points based on depth consistency
and grazing angle checks across multiple camera views.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np
import pycolmap
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numba


@dataclass
class FloaterFilterConfig:
    """Configuration for floater point filtering."""

    vote_threshold: int = 4
    """Number of votes required to remove a 'floater' point."""

    depth_threshold: float = 0.9
    """Threshold to identify a floater (projected_depth < T * refined_depth)."""

    grazing_angle_threshold: float = 0.052
    """Threshold for grazing angle filtering (cos of angle, ~87 degrees)."""

    use_parallel: bool = True
    """Enable parallel processing for faster filtering."""
    
    num_threads: int = 4
    """Number of threads for parallel processing."""
    
    batch_size: int = 50000
    """Number of points to process in each batch for memory efficiency."""

    verbose: int = 0
    """Verbosity level (0=quiet, 1=info, 2=debug)."""


def project_points(
    points3d: np.ndarray, image: pycolmap.Image, camera: pycolmap.Camera
) -> tuple[np.ndarray, np.ndarray]:
    """
    Projects 3D points to the image plane for a given camera.

    Args:
        points3d: Nx3 array of 3D world coordinates
        image: pycolmap Image with camera pose
        camera: pycolmap Camera with intrinsics

    Returns:
        Tuple of (points2d, depths) where:
        - points2d: Nx2 array of 2D image coordinates
        - depths: N array of depth values
    """
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


@numba.jit(nopython=True, parallel=True, cache=True)
def compute_inconsistency_votes_numba(
    points: np.ndarray,
    normals: np.ndarray,
    cam_center: np.ndarray,
    cam_from_world: np.ndarray,
    K: np.ndarray,
    refined_depth: np.ndarray,
    depth_threshold: float,
    grazing_threshold: float,
) -> np.ndarray:
    """
    Numba-accelerated computation of inconsistency votes for a single camera.
    
    This function processes all points in parallel using Numba's JIT compilation
    for significant speedup over pure NumPy operations.
    """
    num_points = len(points)
    h, w = refined_depth.shape
    votes = np.zeros(num_points, dtype=np.int32)
    
    # Transform points to camera coordinates
    points_h = np.ones((num_points, 4))
    points_h[:, :3] = points
    
    for i in numba.prange(num_points):
        # World to camera transformation
        point_cam = cam_from_world @ points_h[i]
        depth = point_cam[2]
        
        if depth <= 0:
            continue
            
        # Project to image plane
        point_cam_norm = point_cam[:3] / depth
        point_2d = K @ point_cam_norm
        u, v = point_2d[0], point_2d[1]
        
        # Check bounds
        if u < 0 or u >= w or v < 0 or v >= h:
            continue
            
        # Check grazing angle
        viewing_dir = points[i] - cam_center
        viewing_norm = np.linalg.norm(viewing_dir)
        if viewing_norm > 0:
            viewing_dir = viewing_dir / viewing_norm
            dot_product = -np.dot(normals[i], viewing_dir)
            if dot_product <= grazing_threshold:
                continue
        
        # Sample depth and check consistency
        u_int, v_int = int(u), int(v)
        ref_depth = refined_depth[v_int, u_int]
        
        if ref_depth > 0 and depth < depth_threshold * ref_depth:
            votes[i] = 1
    
    return votes


class FloaterFilter:
    """
    Filters floating points from densified point clouds using multi-view consistency.

    The filter works by:
    1. Projecting points to each camera view
    2. Checking depth consistency with the refined depth map
    3. Filtering points at grazing angles
    4. Voting to identify floaters across multiple views
    """

    def __init__(
        self,
        config: FloaterFilterConfig | None = None,
        vote_threshold: int | None = None,
        depth_threshold: float | None = None,
        grazing_angle_threshold: float | None = None,
        use_parallel: bool | None = None,
        num_threads: int | None = None,
        batch_size: int | None = None,
        verbose: int | None = None,
    ):
        """
        Initialize the floater filter.

        Args:
            config: Configuration object (optional)
            vote_threshold: Number of inconsistent views to mark as floater
            depth_threshold: Depth consistency threshold
            grazing_angle_threshold: Grazing angle threshold (cosine)
            verbose: Verbosity level
        """
        # Use provided config or create default
        config = config or FloaterFilterConfig()

        # Override config with any provided parameters
        self.vote_threshold = (
            vote_threshold if vote_threshold is not None else config.vote_threshold
        )
        self.depth_threshold = (
            depth_threshold if depth_threshold is not None else config.depth_threshold
        )
        self.grazing_angle_threshold = (
            grazing_angle_threshold
            if grazing_angle_threshold is not None
            else config.grazing_angle_threshold
        )
        self.use_parallel = use_parallel if use_parallel is not None else config.use_parallel
        self.num_threads = num_threads if num_threads is not None else config.num_threads
        self.batch_size = batch_size if batch_size is not None else config.batch_size
        self.verbose = verbose if verbose is not None else config.verbose

    def _process_camera_view(
        self,
        data: dict[str, Any],
        points: np.ndarray,
        normals: np.ndarray,
    ) -> np.ndarray:
        """
        Process a single camera view and return inconsistency votes.
        
        This is separated out for parallel processing.
        """
        refined_depth = data["refined_depth"]
        camera = data["camera"]
        image = data["image"]
        cam_center = data["cam_center"]
        
        h, w = refined_depth.shape
        num_points = len(points)
        votes = np.zeros(num_points, dtype=np.int32)
        
        # Try to use Numba acceleration if matrices are available
        try:
            cam_from_world = image.cam_from_world().matrix()
            K = camera.calibration_matrix()
            
            # Process in batches for memory efficiency
            for i in range(0, num_points, self.batch_size):
                end_idx = min(i + self.batch_size, num_points)
                batch_points = points[i:end_idx]
                batch_normals = normals[i:end_idx]
                
                # Use optimized batch processing
                batch_votes = self._process_batch_vectorized(
                    batch_points, batch_normals, 
                    cam_center, cam_from_world, K,
                    refined_depth, h, w
                )
                votes[i:end_idx] = batch_votes
                
        except Exception:
            # Fallback to original method if Numba fails
            votes = self._process_camera_view_fallback(
                data, points, normals
            )
            
        return votes
    
    def _process_batch_vectorized(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        cam_center: np.ndarray,
        cam_from_world: np.ndarray,
        K: np.ndarray,
        refined_depth: np.ndarray,
        h: int,
        w: int,
    ) -> np.ndarray:
        """
        Optimized vectorized batch processing.
        """
        # Transform all points at once
        points_h = np.hstack([points, np.ones((len(points), 1))])
        points_cam_h = (cam_from_world @ points_h.T).T
        points_cam = points_cam_h[:, :3]
        depths = points_cam[:, 2]
        
        # Early filtering of invalid depths
        valid_mask = depths > 0
        if not np.any(valid_mask):
            return np.zeros(len(points), dtype=np.int32)
        
        # Project valid points
        points_cam_valid = points_cam[valid_mask]
        depths_valid = depths[valid_mask]
        points_cam_norm = points_cam_valid / depths_valid[:, np.newaxis]
        points_2d = (K @ points_cam_norm.T).T[:, :2]
        
        # Bounds check
        u, v = points_2d[:, 0], points_2d[:, 1]
        bounds_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        # Grazing angle check (vectorized)
        viewing_dirs = points[valid_mask] - cam_center
        viewing_norms = np.linalg.norm(viewing_dirs, axis=1)
        viewing_dirs = viewing_dirs / (viewing_norms[:, np.newaxis] + 1e-8)
        dot_products = np.einsum("ij,ij->i", normals[valid_mask], -viewing_dirs)
        grazing_mask = dot_products > self.grazing_angle_threshold
        
        # Combined mask
        final_mask = bounds_mask & grazing_mask
        
        # Initialize result
        votes = np.zeros(len(points), dtype=np.int32)
        
        if np.any(final_mask):
            # Get valid indices
            valid_indices = np.where(valid_mask)[0][final_mask]
            
            # Sample depths
            u_int = u[final_mask].astype(np.int32)
            v_int = v[final_mask].astype(np.int32)
            ref_depths = refined_depth[v_int, u_int]
            proj_depths = depths_valid[final_mask]
            
            # Check consistency
            valid_ref = ref_depths > 0
            inconsistent = proj_depths[valid_ref] < self.depth_threshold * ref_depths[valid_ref]
            
            # Set votes
            inconsistent_indices = valid_indices[valid_ref][inconsistent]
            votes[inconsistent_indices] = 1
            
        return votes
    
    def _process_camera_view_fallback(
        self,
        data: dict[str, Any],
        points: np.ndarray,
        normals: np.ndarray,
    ) -> np.ndarray:
        """
        Fallback method using original implementation.
        """
        refined_depth = data["refined_depth"]
        camera = data["camera"]
        image = data["image"]
        cam_center = data["cam_center"]
        
        h, w = refined_depth.shape
        num_points = len(points)
        votes = np.zeros(num_points, dtype=np.int32)
        
        # Project points to this view
        points2d, depths = project_points(points, image, camera)
        
        # Check grazing angles
        viewing_dirs = points - cam_center
        viewing_norms = np.linalg.norm(viewing_dirs, axis=1)
        viewing_dirs = viewing_dirs / (viewing_norms[:, np.newaxis] + 1e-8)
        
        # Dot product between normal and viewing direction
        dot_products = np.einsum("ij,ij->i", normals, -viewing_dirs)
        not_grazing_mask = dot_products > self.grazing_angle_threshold
        
        u, v = points2d[:, 0], points2d[:, 1]
        
        # Check bounds and validity
        mask_in_bounds = (
            (u >= 0)
            & (u < w)
            & (v >= 0)
            & (v < h)
            & (depths > 0)
            & not_grazing_mask
        )
        
        if np.any(mask_in_bounds):
            # Sample depth values at projected locations
            u_valid = u[mask_in_bounds].astype(np.int32)
            v_valid = v[mask_in_bounds].astype(np.int32)
            
            projected_depths = depths[mask_in_bounds]
            refined_depths_at_proj = refined_depth[v_valid, u_valid]
            
            # Check for valid depth samples
            valid_depth_mask = refined_depths_at_proj > 0
            
            # Check depth consistency
            inconsistent_mask = (
                projected_depths[valid_depth_mask]
                < self.depth_threshold * refined_depths_at_proj[valid_depth_mask]
            )
            
            # Update votes
            indices_in_bounds = np.where(mask_in_bounds)[0]
            indices_with_valid_depth = indices_in_bounds[valid_depth_mask]
            inconsistent_indices = indices_with_valid_depth[inconsistent_mask]
            
            votes[inconsistent_indices] = 1
            
        return votes
    
    def filter_points(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        camera_data: list[dict[str, Any]],
        show_progress: bool = True,
    ) -> tuple[np.ndarray, int]:
        """
        Filter floating points using multi-view consistency.

        Args:
            points: Nx3 array of 3D points
            normals: Nx3 array of point normals
            camera_data: List of camera data dictionaries containing:
                - 'refined_depth': HxW depth map
                - 'camera': pycolmap Camera object
                - 'image': pycolmap Image object
                - 'cam_center': Camera center position
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (mask, num_removed) where:
            - mask: Boolean array indicating points to keep
            - num_removed: Number of points removed
        """
        num_points = len(points)
        if num_points == 0:
            return np.array([], dtype=bool), 0

        # Initialize vote counter
        floater_votes = np.zeros(num_points, dtype=np.int32)

        if self.use_parallel and len(camera_data) > 1:
            # Parallel processing of camera views
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Submit all camera processing tasks
                futures = [
                    executor.submit(self._process_camera_view, data, points, normals)
                    for data in camera_data
                ]
                
                # Collect results with progress bar
                iterator = (
                    tqdm(futures, desc="Filtering Points") if show_progress else futures
                )
                
                for future in iterator:
                    votes = future.result()
                    floater_votes += votes
        else:
            # Sequential processing (original method)
            iterator = (
                tqdm(camera_data, desc="Filtering Points") if show_progress else camera_data
            )
            
            for data in iterator:
                votes = self._process_camera_view(data, points, normals)
                floater_votes += votes

        # Create mask for points to keep
        points_to_keep_mask = floater_votes < self.vote_threshold
        num_removed = np.sum(~points_to_keep_mask)

        if self.verbose > 0:
            removal_percentage = (
                (num_removed / num_points * 100) if num_points > 0 else 0
            )
            print(
                f"[FloaterFilter] Removed {num_removed} points ({removal_percentage:.2f}%)"
            )
            if self.verbose > 1:
                max_votes = np.max(floater_votes) if len(floater_votes) > 0 else 0
                print(f"[FloaterFilter] Max floater votes: {max_votes}")
                vote_histogram = np.bincount(floater_votes)
                for votes, count in enumerate(vote_histogram):
                    if count > 0:
                        print(f"  {votes} votes: {count} points")

        return points_to_keep_mask, num_removed

    def prepare_camera_data(
        self,
        cached_refinement_data: dict[int, dict[str, Any]],
        camera_matrices_cache: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Prepare camera data for filtering from cached data structures.

        This is a helper method to convert the cached data from the pipeline
        into the format expected by filter_points.

        Args:
            cached_refinement_data: Dictionary mapping image_id to refinement data
            camera_matrices_cache: Dictionary mapping image_id to camera matrices

        Returns:
            List of camera data dictionaries ready for filtering
        """
        camera_data = []

        for data in cached_refinement_data.values():
            image_id = data["image"].image_id
            cached_matrices = camera_matrices_cache[image_id]

            # Get the rescaled camera
            camera = data["camera"]

            camera_data.append(
                {
                    "refined_depth": data["refined_depth"],
                    "camera": camera,
                    "image": data["image"],
                    "cam_center": cached_matrices["cam_center"],
                }
            )

        return camera_data
