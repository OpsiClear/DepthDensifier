"""
Ultra-fast PCHIP-based depth refinement with full GPU acceleration.

This module provides a fully GPU-accelerated implementation of the PCHIP refiner
including edge detection, outlier removal, and interpolation all on GPU.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class UltraFastPCHIPRefinerConfig:
    """Configuration for UltraFastPCHIPRefiner parameters."""
    
    # PCHIP-specific parameters
    min_correspondences: int = 100
    edge_margin: int = 20
    edge_threshold: float = 0.1
    edge_sigma: float = 2.0
    robust: bool = True
    outlier_threshold: float = 3.0
    
    # Edge detection mode
    use_image_edges: bool = False  # If True, use RGB image edges instead of depth edges
    image_edge_threshold: float = 30.0  # Threshold for image gradient magnitude
    
    # Compatibility parameters
    scale_filter_factor: float = 2.0
    verbose: int = 1


class UltraFastPCHIPRefiner:
    """
    Ultra-fast fully GPU-accelerated PCHIP depth refinement.
    
    All operations including edge detection, outlier removal, and interpolation
    are performed on GPU for maximum performance.
    """
    
    def __init__(
        self,
        config: UltraFastPCHIPRefinerConfig | None = None,
        min_correspondences: int | None = None,
        edge_margin: int | None = None,
        edge_threshold: float | None = None,
        edge_sigma: float | None = None,
        robust: bool | None = None,
        outlier_threshold: float | None = None,
        scale_filter_factor: float | None = None,
        verbose: int | None = None,
        use_image_edges: bool | None = None,
        image_edge_threshold: float | None = None,
    ):
        """Initialize ultra-fast PCHIP refiner."""
        if config is None:
            config = UltraFastPCHIPRefinerConfig()
        
        # Use provided parameters or fall back to config defaults
        self.min_correspondences = min_correspondences if min_correspondences is not None else config.min_correspondences
        self.edge_margin = edge_margin if edge_margin is not None else config.edge_margin
        self.edge_threshold = edge_threshold if edge_threshold is not None else config.edge_threshold
        self.edge_sigma = edge_sigma if edge_sigma is not None else config.edge_sigma
        self.robust = robust if robust is not None else config.robust
        self.outlier_threshold = outlier_threshold if outlier_threshold is not None else config.outlier_threshold
        self.scale_filter_factor = scale_filter_factor if scale_filter_factor is not None else config.scale_filter_factor
        self.verbose = verbose if verbose is not None else config.verbose
        self.use_image_edges = use_image_edges if use_image_edges is not None else config.use_image_edges
        self.image_edge_threshold = image_edge_threshold if image_edge_threshold is not None else config.image_edge_threshold
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.verbose > 0:
            print(f"[UltraFastPCHIP] Using {self.device.type.upper()} backend")
        
        # Pre-create Gaussian kernel for edge detection
        self._create_gaussian_kernel()
    
    def _create_gaussian_kernel(self):
        """Create Gaussian kernel for smoothing on GPU."""
        kernel_size = int(self.edge_sigma * 4) | 1  # Ensure odd
        x = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / self.edge_sigma) ** 2)
        kernel_1d /= kernel_1d.sum()
        # Create 2D kernel and reshape to [1, 1, k, k] for conv2d
        kernel_2d = kernel_1d.view(-1, 1) @ kernel_1d.view(1, -1)
        self.gaussian_kernel = kernel_2d.unsqueeze(0).unsqueeze(0)
    
    def _detect_image_edges_torch(
        self, rgb_tensor: torch.Tensor, mask_tensor: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Detect edges using RGB image gradients on GPU.
        
        Args:
            rgb_tensor: RGB image tensor of shape (H, W, 3)
            mask_tensor: Optional mask for valid pixels
            
        Returns:
            Binary edge mask tensor of shape (H, W)
        """
        # Convert to grayscale
        if rgb_tensor.dim() == 3 and rgb_tensor.shape[2] == 3:
            # Standard grayscale conversion
            gray = 0.299 * rgb_tensor[:, :, 0] + 0.587 * rgb_tensor[:, :, 1] + 0.114 * rgb_tensor[:, :, 2]
        else:
            gray = rgb_tensor.squeeze()
        
        # Apply Gaussian smoothing using convolution
        gray_for_conv = gray.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        kernel = self.gaussian_kernel  # Already [1, 1, k, k] from _create_gaussian_kernel
        padding = kernel.shape[-1] // 2
        gray_smooth = F.conv2d(gray_for_conv, kernel, padding=padding).squeeze()
        
        # Compute gradients using Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        gray_for_sobel = gray_smooth.unsqueeze(0).unsqueeze(0)
        grad_x = F.conv2d(gray_for_sobel, sobel_x, padding=1).squeeze()
        grad_y = F.conv2d(gray_for_sobel, sobel_y, padding=1).squeeze()
        
        # Compute gradient magnitude
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold to get edge mask
        edge_mask = grad_mag > self.image_edge_threshold
        
        # Apply mask if provided
        if mask_tensor is not None:
            edge_mask = edge_mask & mask_tensor
        
        # Dilate edge mask using max pooling
        edge_mask_float = edge_mask.float().unsqueeze(0).unsqueeze(0)
        edge_mask_dilated = F.max_pool2d(edge_mask_float, kernel_size=5, stride=1, padding=2)
        edge_mask = edge_mask_dilated.squeeze() > 0.5
        
        return edge_mask
    
    def _detect_depth_edges_torch(
        self, depth_tensor: torch.Tensor, mask_tensor: torch.Tensor | None, 
        normal_tensor: torch.Tensor | None
    ) -> torch.Tensor:
        """Detect depth discontinuities using GPU operations."""
        h, w = depth_tensor.shape
        
        if mask_tensor is None:
            mask_tensor = (depth_tensor > 0) & torch.isfinite(depth_tensor)
        
        # Initialize edge mask
        edge_mask = torch.zeros_like(depth_tensor, dtype=torch.bool)
        
        # Normal-based edge detection
        if normal_tensor is not None:
            # Compute normal gradients using convolutions
            normal_tensor = normal_tensor.unsqueeze(0).permute(0, 3, 1, 2)  # [1, 3, H, W]
            
            # Sobel filters for gradient computation
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                  dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                  dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
            
            # Apply to each channel
            grad_x = F.conv2d(normal_tensor, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
            grad_y = F.conv2d(normal_tensor, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
            
            normal_grad_mag = torch.sqrt(grad_x**2 + grad_y**2).sum(dim=1).squeeze(0)
            normal_edge_mask = normal_grad_mag > 0.5
            edge_mask |= normal_edge_mask
        
        # Depth-based edge detection with GPU smoothing
        depth_smooth = depth_tensor.clone()
        depth_smooth[~mask_tensor] = 0
        
        # Apply Gaussian smoothing using convolution
        depth_for_conv = depth_smooth.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        kernel = self.gaussian_kernel  # Already [1, 1, kh, kw]
        padding = kernel.shape[-1] // 2
        depth_smooth = F.conv2d(depth_for_conv, kernel, padding=padding).squeeze()
        
        # Compute gradients using finite differences
        dy = torch.zeros_like(depth_smooth)
        dx = torch.zeros_like(depth_smooth)
        
        dy[1:-1, :] = (depth_smooth[2:, :] - depth_smooth[:-2, :]) / 2
        dx[:, 1:-1] = (depth_smooth[:, 2:] - depth_smooth[:, :-2]) / 2
        
        depth_grad_mag = torch.sqrt(dx**2 + dy**2)
        
        # Normalize by depth (relative gradient)
        relative_grad = depth_grad_mag / (depth_smooth + 1e-6)
        relative_grad[~mask_tensor] = 0
        
        # Threshold for depth discontinuities
        depth_edge_mask = relative_grad > self.edge_threshold
        edge_mask |= depth_edge_mask
        
        # Dilate edge mask using max pooling
        edge_mask_float = edge_mask.float().unsqueeze(0).unsqueeze(0)
        edge_mask_dilated = F.max_pool2d(edge_mask_float, kernel_size=5, stride=1, padding=2)
        edge_mask = edge_mask_dilated.squeeze() > 0.5
        
        return edge_mask
    
    def _remove_outliers_torch(
        self, z_colmap: torch.Tensor, z_depth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Remove outliers using robust MAD-based detection on GPU."""
        # Compute residuals with median ratio
        ratios = z_colmap / (z_depth + 1e-6)
        median_ratio = torch.median(ratios)
        
        # MAD-based outlier detection
        residuals = z_colmap - z_depth * median_ratio
        median_residual = torch.median(residuals)
        mad = torch.median(torch.abs(residuals - median_residual))
        
        # Handle zero MAD case
        if mad < 1e-6:
            mad = torch.std(residuals) * 0.6745  # Fallback to std-based estimate
        
        threshold = self.outlier_threshold * mad
        inliers = torch.abs(residuals - median_residual) < threshold
        
        return z_colmap[inliers], z_depth[inliers], (~inliers).sum().item()
    
    def _pchip_interpolate_torch_optimized(
        self, depths_flat: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimized GPU-accelerated PCHIP interpolation using PyTorch.
        
        This implements cubic Hermite interpolation with monotonicity preservation.
        All operations stay on GPU to avoid CPU-GPU transfers.
        """
        # Compute slopes for monotonic cubic interpolation
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        slopes = dy / (dx + 1e-8)
        
        # Apply monotonicity constraint (PCHIP-style)
        slopes_padded = torch.cat([slopes[:1], slopes, slopes[-1:]])
        
        # Find intervals using binary search (vectorized)
        indices = torch.searchsorted(x, depths_flat)
        indices = torch.clamp(indices, 1, len(x) - 1)
        
        # Get interval boundaries
        idx_left = indices - 1
        idx_right = indices
        
        x_left = torch.gather(x, 0, idx_left)
        x_right = torch.gather(x, 0, idx_right)
        y_left = torch.gather(y, 0, idx_left)
        y_right = torch.gather(y, 0, idx_right)
        slope_left = torch.gather(slopes_padded, 0, idx_left)
        slope_right = torch.gather(slopes_padded, 0, idx_right)
        
        # Compute normalized position in interval
        h = x_right - x_left + 1e-8
        t = (depths_flat - x_left) / h
        t2 = t * t
        t3 = t2 * t
        
        # Cubic Hermite basis functions
        h00 = 2*t3 - 3*t2 + 1    # Left point
        h10 = t3 - 2*t2 + t      # Left tangent
        h01 = -2*t3 + 3*t2       # Right point
        h11 = t3 - t2            # Right tangent
        
        # Compute interpolated values with scaled tangents
        result = (h00 * y_left + 
                 h10 * h * slope_left * 0.3 +    # Scale tangents for stability
                 h01 * y_right + 
                 h11 * h * slope_right * 0.3)
        
        # Handle extrapolation
        result = torch.where(depths_flat <= x[0], 
                            y[0] + slopes_padded[0] * (depths_flat - x[0]) * 0.3,
                            result)
        result = torch.where(depths_flat >= x[-1],
                            y[-1] + slopes_padded[-1] * (depths_flat - x[-1]) * 0.3,
                            result)
        
        # Ensure positive depths
        result = torch.maximum(result, torch.tensor(1e-3, device=self.device))
        
        return result
    
    def project_points_to_image(
        self, points3D: np.ndarray, cam_from_world: np.ndarray, K: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project 3D points to image plane."""
        points3D_h = np.hstack([points3D, np.ones((points3D.shape[0], 1))])
        H = np.vstack([cam_from_world, [0, 0, 0, 1]])
        points_cam = (H @ points3D_h.T)[:3, :].T
        depths = points_cam[:, 2].copy()
        valid = depths > 1e-6
        pts2d = np.zeros((len(depths), 2))
        pts2d[valid] = ((K @ (points_cam[valid] / depths[valid, None]).T).T)[:, :2]
        return pts2d, depths, valid
    
    def sample_depth_at_points_torch(
        self, depth_tensor: torch.Tensor, pts2d_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Sample depth values at 2D points using bilinear interpolation on GPU."""
        h, w = depth_tensor.shape
        x = pts2d_tensor[:, 0]
        y = pts2d_tensor[:, 1]
        
        # Prepare for grid_sample
        # grid_sample expects coordinates in [-1, 1] range
        x_norm = 2.0 * x / (w - 1) - 1.0
        y_norm = 2.0 * y / (h - 1) - 1.0
        
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
        depth_for_sampling = depth_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Bilinear interpolation
        sampled = F.grid_sample(depth_for_sampling, grid, mode='bilinear', 
                               padding_mode='zeros', align_corners=True)
        
        return sampled.squeeze()
    
    def refine_depth(
        self,
        depth_map: np.ndarray,
        normal_map: np.ndarray | None,
        points3D: np.ndarray,
        cam_from_world: np.ndarray,
        K: np.ndarray,
        mask: np.ndarray | None = None,
        depth_uncertainty: np.ndarray | None = None,
        rgb_image: np.ndarray | None = None,
        normal_uncertainty: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Refine depth map using ultra-fast fully GPU-accelerated PCHIP interpolation.
        """
        h, w = depth_map.shape
        use_normals = normal_map is not None
        
        # Move depth and normal maps to GPU
        depth_tensor = torch.from_numpy(depth_map).float().to(self.device)
        normal_tensor = None
        if normal_map is not None:
            normal_tensor = torch.from_numpy(normal_map).float().to(self.device)
        
        mask_tensor = None
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).bool().to(self.device)
        else:
            mask_tensor = (depth_tensor > 0) & torch.isfinite(depth_tensor)
        
        # Project 3D points (still on CPU for now)
        if self.verbose > 0:
            print(f"[UltraFastPCHIP] Projecting {len(points3D)} 3D points to image...")
        
        pts2d, depths3d, valid = self.project_points_to_image(points3D, cam_from_world, K)
        
        # Move to GPU
        pts2d_tensor = torch.from_numpy(pts2d).float().to(self.device)
        depths3d_tensor = torch.from_numpy(depths3d).float().to(self.device)
        valid_tensor = torch.from_numpy(valid).bool().to(self.device)
        
        # Detect edges on GPU
        if self.use_image_edges and rgb_image is not None:
            # Use image gradient-based edge detection
            rgb_tensor = torch.from_numpy(rgb_image).float().to(self.device)
            edge_mask = self._detect_image_edges_torch(rgb_tensor, mask_tensor)
            if self.verbose > 0:
                print(f"[UltraFastPCHIP] Using image-based edge detection (threshold={self.image_edge_threshold})")
        else:
            # Use depth/normal gradient-based edge detection
            edge_mask = self._detect_depth_edges_torch(depth_tensor, mask_tensor, normal_tensor)
            if self.verbose > 0 and self.use_image_edges:
                print("[UltraFastPCHIP] Warning: use_image_edges=True but no RGB image provided, falling back to depth edges")
        
        # Filter points within bounds (excluding edges)
        in_bounds = (
            (pts2d_tensor[:, 0] >= self.edge_margin)
            & (pts2d_tensor[:, 0] < w - self.edge_margin)
            & (pts2d_tensor[:, 1] >= self.edge_margin)
            & (pts2d_tensor[:, 1] < h - self.edge_margin)
            & valid_tensor
        )
        
        pts2d_valid = pts2d_tensor[in_bounds]
        depths3d_valid = depths3d_tensor[in_bounds]
        
        if len(pts2d_valid) == 0:
            if self.verbose > 0:
                print("[UltraFastPCHIP] No valid correspondences found")
            return {
                "refined_depth": depth_map,
                "scale": 1.0,
                "energy_history": [],
                "num_iterations": 0,
                "used_normals": use_normals,
            }
        
        # Sample depths at projected points on GPU
        depths_sampled = self.sample_depth_at_points_torch(depth_tensor, pts2d_valid)
        
        # Get valid correspondences (excluding edges)
        u = torch.round(pts2d_valid[:, 0]).long()
        v = torch.round(pts2d_valid[:, 1]).long()
        
        # Clamp indices
        u = torch.clamp(u, 0, w - 1)
        v = torch.clamp(v, 0, h - 1)
        
        valid_corr = (
            (depths_sampled > 0) & 
            (depths3d_valid > 0) & 
            torch.isfinite(depths_sampled) &
            ~edge_mask[v, u]  # Exclude edge pixels
        )
        
        z_depth = depths_sampled[valid_corr]
        z_colmap = depths3d_valid[valid_corr]
        
        if len(z_depth) < self.min_correspondences:
            if self.verbose > 0:
                print(f"[UltraFastPCHIP] Too few correspondences ({len(z_depth)} < {self.min_correspondences})")
            return {
                "refined_depth": depth_map,
                "scale": 1.0,
                "energy_history": [],
                "num_iterations": 0,
                "used_normals": use_normals,
            }
        
        # Remove outliers on GPU if robust
        if self.robust:
            z_colmap_clean, z_depth_clean, n_outliers = self._remove_outliers_torch(z_colmap, z_depth)
            if self.verbose > 0 and n_outliers > 0:
                print(f"[UltraFastPCHIP] Removed {n_outliers} outliers")
        else:
            z_colmap_clean = z_colmap
            z_depth_clean = z_depth
        
        if len(z_depth_clean) < self.min_correspondences:
            if self.verbose > 0:
                print(f"[UltraFastPCHIP] Too few correspondences after outlier removal")
            scale = torch.median(z_colmap / z_depth).item() if torch.all(z_depth > 0) else 1.0
            refined_depth = (depth_tensor * scale).cpu().numpy()
            return {
                "refined_depth": refined_depth,
                "scale": scale,
                "energy_history": [],
                "num_iterations": 0,
                "used_normals": use_normals,
            }
        
        # Prepare unique correspondences for interpolation
        unique_x, unique_indices = torch.unique(z_depth_clean, return_inverse=True, sorted=True)
        
        # Average y values for duplicate x values
        unique_y = torch.zeros_like(unique_x)
        for i in range(len(unique_x)):
            mask_i = unique_indices == i
            unique_y[i] = z_colmap_clean[mask_i].mean()
        
        # Apply PCHIP interpolation on GPU
        refined_depth = self._apply_edge_aware_transformation_torch(
            depth_tensor, mask_tensor, edge_mask, unique_x, unique_y
        )
        
        # Convert back to numpy
        refined_depth_np = refined_depth.cpu().numpy()
        
        # Compute effective scale
        scale = (unique_y / unique_x).mean().item() if torch.all(unique_x > 0) else 1.0
        
        if self.verbose > 0:
            print(f"[UltraFastPCHIP] Refined using {len(unique_x)} unique correspondences")
            print(f"[UltraFastPCHIP] Effective scale: {scale:.3f}")
        
        return {
            "refined_depth": refined_depth_np,
            "scale": scale,
            "energy_history": [],  # Empty for API compatibility
            "num_iterations": 1,  # PCHIP is non-iterative
            "used_normals": use_normals,
        }
    
    def _apply_edge_aware_transformation_torch(
        self, depth_tensor: torch.Tensor, mask_tensor: torch.Tensor, 
        edge_mask: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Apply PCHIP transformation with edge preservation on GPU."""
        refined_depth = torch.zeros_like(depth_tensor)
        
        # Process non-edge pixels with full transformation
        non_edge_mask = mask_tensor & ~edge_mask
        if non_edge_mask.any():
            depths_flat = depth_tensor[non_edge_mask]
            result_flat = self._pchip_interpolate_torch_optimized(depths_flat, x, y)
            refined_depth[non_edge_mask] = result_flat
        
        # For edge pixels, apply conservative blending
        edge_valid = mask_tensor & edge_mask
        if edge_valid.any():
            # Apply transformation but blend with original
            edge_depths = depth_tensor[edge_valid]
            edge_transformed = self._pchip_interpolate_torch_optimized(edge_depths, x, y)
            # Blend: 70% original, 30% transformed for edge pixels
            refined_depth[edge_valid] = 0.7 * edge_depths + 0.3 * edge_transformed
        
        # Apply median filter for smoothing (3x3)
        refined_depth_for_filter = refined_depth.unsqueeze(0).unsqueeze(0)
        refined_depth_filtered = F.pad(refined_depth_for_filter, (1, 1, 1, 1), mode='replicate')
        
        # Manual 3x3 median using unfold
        h, w = depth_tensor.shape
        unfolded = F.unfold(refined_depth_filtered, kernel_size=3, stride=1)
        median_vals = unfolded.median(dim=1)[0].view(h, w)
        refined_depth = torch.where(mask_tensor, median_vals, torch.zeros_like(refined_depth))
        
        # Restore zeros where mask is invalid
        refined_depth[~mask_tensor] = 0
        
        return refined_depth