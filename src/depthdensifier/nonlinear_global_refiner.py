"""
Nonlinear global depth refinement using polynomial or spline transformations.

This refiner fits a global nonlinear function to map MoGe depths to COLMAP depths,
capturing complex relationships while avoiding local overfitting.
"""

import numpy as np
import torch
from scipy import optimize
from scipy.interpolate import UnivariateSpline, PchipInterpolator
import pycolmap
from typing import Optional, Dict, Any, Literal


class NonlinearGlobalRefiner:
    """
    Global nonlinear depth refinement using various function fitting methods.
    
    Supports multiple transformation types:
    - Polynomial: d_refined = sum(a_i * d^i) for i=0 to degree
    - Spline: Smooth spline interpolation with controlled knots
    - PCHIP: Piecewise Cubic Hermite Interpolating Polynomial (monotonic)
    - Power: d_refined = a * d^b + c (power law transformation)
    """
    
    def __init__(self, 
                 method: Literal['polynomial', 'spline', 'pchip', 'power'] = 'spline',
                 degree: int = 3,
                 n_knots: int = 7,
                 min_corr: int = 100,
                 edge_margin: int = 20,
                 robust: bool = True,
                 outlier_threshold: float = 3.0,
                 verbose: int = 1):
        """
        :param method: Type of nonlinear transformation
        :param degree: Polynomial degree (for polynomial method)
        :param n_knots: Number of knots for spline fitting
        :param min_corr: Minimum correspondences required
        :param edge_margin: Pixels to exclude from edges
        :param robust: Use robust fitting with outlier rejection
        :param outlier_threshold: MAD threshold for outlier detection
        :param verbose: Verbosity level
        """
        self.method = method
        self.degree = degree
        self.n_knots = n_knots
        self.min_corr = min_corr
        self.edge_margin = edge_margin
        self.robust = robust
        self.outlier_threshold = outlier_threshold
        self.verbose = verbose
    
    def refine(self,
               depth_map: torch.Tensor,
               mask: torch.Tensor,
               points3D_world: np.ndarray,
               image: pycolmap.Image,
               camera: pycolmap.Camera,
               normal_map: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Refine depth using global nonlinear transformation.
        
        :param depth_map: Input depth map (H, W)
        :param mask: Valid mask (H, W)
        :param points3D_world: COLMAP 3D points
        :param image: COLMAP image with pose
        :param camera: COLMAP camera with intrinsics
        :param normal_map: Optional normal map (unused)
        :return: Dictionary with refined_depth and transformation parameters
        """
        device = depth_map.device
        dtype = depth_map.dtype
        H, W = depth_map.shape
        
        # Get correspondences (excluding edges)
        u, v, z_colmap, z_moge = self._get_correspondences(
            depth_map, mask, points3D_world, image, camera
        )
        
        if len(u) < self.min_corr:
            if self.verbose:
                print(f"[NonlinearGlobal] Too few correspondences ({len(u)} < {self.min_corr})")
            return {"refined_depth": depth_map, "transform_params": None}
        
        # Remove outliers if robust mode
        if self.robust:
            z_colmap, z_moge, n_outliers = self._remove_outliers(z_colmap, z_moge)
            if self.verbose and n_outliers > 0:
                print(f"[NonlinearGlobal] Removed {n_outliers} outliers")
        
        # Fit nonlinear transformation
        if self.method == 'polynomial':
            transform_func, params = self._fit_polynomial(z_moge, z_colmap)
        elif self.method == 'spline':
            transform_func, params = self._fit_spline(z_moge, z_colmap)
        elif self.method == 'pchip':
            transform_func, params = self._fit_pchip(z_moge, z_colmap)
        elif self.method == 'power':
            transform_func, params = self._fit_power_law(z_moge, z_colmap)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Compute RMSE
        if self.verbose:
            z_refined_corr = transform_func(z_moge)
            rmse_before = np.sqrt(np.mean((z_colmap - z_moge) ** 2))
            rmse_after = np.sqrt(np.mean((z_colmap - z_refined_corr) ** 2))
            
            # Compute equivalent linear scale for reference
            linear_scale = np.mean(z_colmap / z_moge) if np.all(z_moge > 0) else 1.0
            
            print(f"[NonlinearGlobal] Method: {self.method}, {len(z_moge)} correspondences")
            print(f"[NonlinearGlobal] RMSE: {rmse_before:.3f} -> {rmse_after:.3f}")
            print(f"[NonlinearGlobal] Equivalent linear scale: ~{linear_scale:.3f}")
        
        # Apply transformation to entire depth map
        depth_np = depth_map.cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        # Apply transformation only to valid pixels
        refined_np = np.zeros_like(depth_np)
        valid_pixels = mask_np & (depth_np > 0) & np.isfinite(depth_np)
        
        if valid_pixels.any():
            # Apply in chunks for memory efficiency
            valid_depths = depth_np[valid_pixels]
            
            # Clip to training range to avoid extrapolation issues
            min_train, max_train = z_moge.min(), z_moge.max()
            valid_depths_clipped = np.clip(valid_depths, min_train, max_train)
            
            # Apply transformation
            refined_valid = transform_func(valid_depths_clipped)
            
            # Ensure positive depths
            refined_valid = np.maximum(refined_valid, 0.001)
            
            refined_np[valid_pixels] = refined_valid
        
        refined_depth = torch.from_numpy(refined_np).to(device).to(dtype)
        
        return {
            "refined_depth": refined_depth,
            "method": self.method,
            "transform_params": params,
            "num_correspondences": len(z_moge),
            "training_range": (float(z_moge.min()), float(z_moge.max()))
        }
    
    def _get_correspondences(self, depth_map, mask, points3D_world, image, camera):
        """Extract correspondences between COLMAP and depth map, excluding edges."""
        H, W = depth_map.shape
        
        # Project COLMAP points
        points3D_world = np.atleast_2d(points3D_world)
        if points3D_world.shape[0] == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Get camera parameters
        cam_from_world = image.cam_from_world().matrix()[:3, :]
        K = camera.calibration_matrix()
        
        # Transform to camera space
        points_cam = (cam_from_world[:3, :3] @ points3D_world.T + 
                     cam_from_world[:3, 3:4]).T
        
        # Get depths
        z_colmap = points_cam[:, 2]
        
        # Project to image
        points_2d = (K @ points_cam.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]
        
        # Round to pixel coordinates
        u = np.round(points_2d[:, 0]).astype(int)
        v = np.round(points_2d[:, 1]).astype(int)
        
        # Filter valid projections - exclude edges
        margin = self.edge_margin
        valid = (u >= margin) & (u < W - margin) & (v >= margin) & (v < H - margin) & (z_colmap > 0)
        u = u[valid]
        v = v[valid]
        z_colmap = z_colmap[valid]
        
        # Sample MoGe depths
        depth_np = depth_map.cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        z_moge = depth_np[v, u]
        valid_mask = mask_np[v, u] & (z_moge > 0) & np.isfinite(z_moge)
        
        return u[valid_mask], v[valid_mask], z_colmap[valid_mask], z_moge[valid_mask]
    
    def _remove_outliers(self, z_colmap, z_moge):
        """Remove outliers using MAD (Median Absolute Deviation)."""
        # Compute residuals with simple linear fit
        ratios = z_colmap / (z_moge + 1e-6)
        median_ratio = np.median(ratios)
        
        # MAD-based outlier detection
        residuals = z_colmap - z_moge * median_ratio
        mad = np.median(np.abs(residuals - np.median(residuals)))
        threshold = self.outlier_threshold * mad
        
        inliers = np.abs(residuals - np.median(residuals)) < threshold
        
        return z_colmap[inliers], z_moge[inliers], (~inliers).sum()
    
    def _fit_polynomial(self, x, y):
        """Fit polynomial transformation."""
        # Fit polynomial of specified degree
        coeffs = np.polyfit(x, y, self.degree)
        
        def transform(depths):
            return np.polyval(coeffs, depths)
        
        return transform, {"coefficients": coeffs.tolist()}
    
    def _fit_spline(self, x, y):
        """Fit smooth spline transformation."""
        # Sort data for spline fitting
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        # Remove duplicates by averaging y values for duplicate x
        unique_x, inverse = np.unique(x_sorted, return_inverse=True)
        unique_y = np.zeros(len(unique_x))
        for i, xi in enumerate(unique_x):
            mask = x_sorted == xi
            unique_y[i] = np.mean(y_sorted[mask])
        
        # Ensure we have enough unique points
        if len(unique_x) < 4:
            # Fall back to linear if not enough points
            coeffs = np.polyfit(x, y, 1)
            def transform(depths):
                return np.polyval(coeffs, depths)
            return transform, {"n_knots": 2, "degree": 1, "smoothing": 0}
        
        # Fit smooth spline with automatic knot selection
        # Use a more conservative smoothing factor
        smoothing = None  # Let scipy determine optimal smoothing
        k = min(3, len(unique_x)-1)  # Spline degree
        
        try:
            spline = UnivariateSpline(unique_x, unique_y, k=k, s=smoothing)
            
            # Get the actual knots used by the spline
            knots = spline.get_knots()
            
            def transform(depths):
                # Handle extrapolation properly
                result = spline(depths)
                # Replace any NaN values with linear extrapolation
                if np.any(np.isnan(result)):
                    # Use linear fit for extrapolation
                    coeffs = np.polyfit(x, y, 1)
                    nan_mask = np.isnan(result)
                    result[nan_mask] = np.polyval(coeffs, depths[nan_mask])
                return result
            
            return transform, {"n_knots": len(knots),
                              "degree": k,
                              "smoothing": spline.get_residual()}
        except Exception as e:
            # Fallback to polynomial if spline fails
            if self.verbose:
                print(f"[NonlinearGlobal] Spline fitting failed: {e}, using polynomial")
            coeffs = np.polyfit(x, y, min(3, len(x) - 1))
            def transform(depths):
                return np.polyval(coeffs, depths)
            return transform, {"coefficients": coeffs.tolist()}
    
    def _fit_pchip(self, x, y):
        """Fit PCHIP (monotonic) interpolation."""
        # Sort and remove duplicates
        unique_x, inverse = np.unique(x, return_inverse=True)
        
        # Average y values for duplicate x
        unique_y = np.zeros(len(unique_x))
        for i, xi in enumerate(unique_x):
            mask = x == xi
            unique_y[i] = np.mean(y[mask])
        
        # Create PCHIP interpolator
        pchip = PchipInterpolator(unique_x, unique_y)
        
        def transform(depths):
            # Extrapolate linearly outside training range
            result = pchip(depths, extrapolate=False)
            
            # Linear extrapolation for out-of-range values
            below_mask = depths < unique_x[0]
            above_mask = depths > unique_x[-1]
            
            if below_mask.any():
                slope_start = (unique_y[1] - unique_y[0]) / (unique_x[1] - unique_x[0])
                result[below_mask] = unique_y[0] + slope_start * (depths[below_mask] - unique_x[0])
            
            if above_mask.any():
                slope_end = (unique_y[-1] - unique_y[-2]) / (unique_x[-1] - unique_x[-2])
                result[above_mask] = unique_y[-1] + slope_end * (depths[above_mask] - unique_x[-1])
            
            return result
        
        return transform, {"n_control_points": len(unique_x)}
    
    def _fit_power_law(self, x, y):
        """Fit power law transformation: y = a * x^b + c."""
        # Initial guess: linear relationship
        initial_guess = [1.0, 1.0, 0.0]  # a, b, c
        
        def power_func(x, a, b, c):
            return a * np.power(x + 1e-6, b) + c
        
        def objective(params):
            a, b, c = params
            pred = power_func(x, a, b, c)
            return np.sum((y - pred) ** 2)
        
        # Bounds: a > 0, b > 0, c can be any
        bounds = [(0.1, 10), (0.1, 3), (-10, 10)]
        
        result = optimize.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            a, b, c = result.x
        else:
            # Fallback to linear
            a, b, c = 1.0, 1.0, 0.0
            if self.verbose:
                print("[NonlinearGlobal] Power law fitting failed, using linear")
        
        def transform(depths):
            return power_func(depths, a, b, c)
        
        return transform, {"a": float(a), "b": float(b), "c": float(c),
                          "equation": f"d' = {a:.3f} * d^{b:.3f} + {c:.3f}"}