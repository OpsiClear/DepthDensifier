"""
DepthDensifier Gaussian Initialization Module

This module provides functionality for initializing 2D Gaussians for image reconstruction
and depth densification tasks. It supports gradient-based and random initialization strategies
to place Gaussians at optimal locations in the image space.

Key Features:
    - Gradient-based initialization: Places Gaussians at high-gradient regions
    - Random initialization: Uniform distribution across the image
    - Color initialization: Samples initial RGB features from target images
    - Flexible configuration: Customizable parameters for different use cases

Main Functions:
    - initialize_gaussians(): Primary function for Gaussian initialization
    - compute_gradient_map(): Computes gradient-based probability distributions
    - sample_positions(): Samples Gaussian positions based on probability maps
    - load_images(): Loads and preprocesses target images

Usage Example:
    # Load target image
    images, _, _ = load_images("image.jpg", gamma=2.2)
    gt_images = torch.from_numpy(images).to("cuda")
    img_h, img_w = gt_images.shape[1], gt_images.shape[2]
    
    # Initialize Gaussians with gradient-based placement
    gaussians = initialize_gaussians(
        gt_images=gt_images,
        num_gaussians=10000,
        img_h=img_h,
        img_w=img_w,
        init_mode="gradient",
        init_random_ratio=0.3,
        init_scale=5.0,
        gamma=2.2,
        device="cuda"
    )
    
    # Access initialized parameters
    positions = gaussians["xy"]      # [N, 2] - Gaussian centers
    scales = gaussians["scale"]      # [N, 2] - Gaussian scales  
    rotations = gaussians["rot"]     # [N, 1] - Gaussian rotations
    features = gaussians["feat"]     # [N, C] - RGB features

Author: DepthDensifier Team
License: See LICENSE file
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import tyro


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

ALLOWED_IMAGE_FILE_FORMATS: list[str] = [".jpeg", ".jpg", ".png"]
ALLOWED_IMAGE_TYPES: dict[str, int] = {"RGB": 3, "RGBA": 3, "L": 1}


def get_grid(
    h: int,
    w: int,
    x_lim: np.ndarray = np.asarray([0, 1]),
    y_lim: np.ndarray = np.asarray([0, 1]),
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a 2D grid of coordinates (optimized with device support)."""
    x = torch.linspace(x_lim[0], x_lim[1], steps=w + 1, device=device, dtype=dtype)[:-1] + 0.5 / w
    y = torch.linspace(y_lim[0], y_lim[1], steps=h + 1, device=device, dtype=dtype)[:-1] + 0.5 / h
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid


def to_output_format(
    image: np.ndarray | torch.Tensor, gamma: float | None
) -> np.ndarray:
    """Convert image to output format for saving."""
    if len(image.shape) not in [2, 3]:
        raise ValueError(f"Wrong image format: shape = {image.shape}")
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().clone().numpy()
    if len(image.shape) == 3 and image.shape[2] not in [1, 3]:
        image = image.transpose(1, 2, 0)
        if image.shape[2] not in [1, 3]:
            raise ValueError(f"Wrong image format: shape = {image.shape}")
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image.squeeze(axis=2)
    image = np.clip(image, 0.0, 1.0)
    if gamma is not None:
        image = np.power(image, 1.0 / gamma)
    image = (255.0 * image).astype(np.uint8)
    return image


def save_image(
    image: np.ndarray | torch.Tensor,
    save_path: str | Path,
    gamma: float | None = None,
    zoom: float | None = None,
) -> None:
    """Save image to disk.
    
    Usage example:
        # Save numpy array
        save_image(gradient_map, "gradient.png", gamma=2.2)
        
        # Save torch tensor with zoom
        save_image(gradient_map, "gradient_zoomed.png", zoom=2.0)
        
        # Save without gamma correction
        save_image(rendered_image, "output.jpg")
    """
    image = to_output_format(image, gamma)
    image = Image.fromarray(image)
    if zoom is not None and zoom > 0.0:
        width, height = image.size
        image = image.resize(
            (round(width * zoom), round(height * zoom)), resample=Image.Resampling.BOX
        )
    image.save(save_path)


def load_images(
    load_path: str | Path,
    downsample_ratio: float | None = None,
    gamma: float | None = None,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Load target images or textures from a directory or a single file.
    
    Usage example:
        # Load single image
        images, channels, filenames = load_images("path/to/image.jpg", gamma=2.2)
        
        # Load images from directory with downsampling
        images, channels, filenames = load_images(
            "path/to/images/", 
            downsample_ratio=2.0, 
            gamma=1.0
        )
        
        # Convert to torch tensor
        gt_images = torch.from_numpy(images).to("cuda")
    """
    image_list = []
    image_path_list = []
    image_fname_list = []
    num_channels_list = []

    if (
        os.path.isfile(load_path)
        and os.path.splitext(load_path)[1].lower() in ALLOWED_IMAGE_FILE_FORMATS
    ):
        image_path_list.append(load_path)
    elif os.path.isdir(load_path):
        for file in sorted(os.listdir(load_path), key=str.lower):
            if os.path.splitext(file)[1].lower() in ALLOWED_IMAGE_FILE_FORMATS:
                image_path_list.append(os.path.join(load_path, file))

    if len(image_path_list) == 0:
        raise FileNotFoundError(f"No supported image file found at '{load_path}'")

    for image_path in image_path_list:
        image_fname_list.append(os.path.splitext(os.path.basename(image_path))[0])
        image = Image.open(image_path)

        if image.mode not in ALLOWED_IMAGE_TYPES:
            raise TypeError(
                f"Only support images of type {list(ALLOWED_IMAGE_TYPES.keys())} in JPEG or PNG format"
            )

        num_channels = ALLOWED_IMAGE_TYPES[image.mode]
        num_channels_list.append(num_channels)

        if downsample_ratio is not None:
            image = image.resize(
                (
                    round(image.width / downsample_ratio),
                    round(image.height / downsample_ratio),
                ),
                resample=Image.Resampling.BILINEAR,
            )

        image = np.asarray(image, dtype=np.float32) / 255.0
        if gamma is not None:
            image = np.power(image, gamma)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        image = image.transpose(2, 0, 1)
        image = image[:num_channels]
        image_list.append(image)

    return np.concatenate(image_list, axis=0), num_channels_list, image_fname_list



# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================


def compute_image_gradients_gpu(image: torch.Tensor) -> torch.Tensor:
    """Compute image gradients using GPU-accelerated operations."""
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    
    # Apply to each channel
    gradients = []
    for i in range(image.shape[0]):
        channel = image[i:i+1].unsqueeze(0)  # [1, 1, H, W]
        gx = F.conv2d(channel, sobel_x, padding=1)
        gy = F.conv2d(channel, sobel_y, padding=1)
        gradient_mag = torch.sqrt(gx**2 + gy**2)
        gradients.append(gradient_mag.squeeze(0))
    
    # Combine channels using L2 norm
    gradient_tensor = torch.stack(gradients, dim=0)  # [C, 1, H, W]
    combined_gradient = torch.norm(gradient_tensor, dim=0).squeeze(0)  # [H, W]
    
    return combined_gradient


def compute_depth_normal_gradient_mask(
    depth_map: torch.Tensor,
    normal_map: torch.Tensor | None = None,
    depth_threshold: float = 0.2,
    normal_threshold: float = 0.3,
    edge_sigma: float = 1.0,
) -> torch.Tensor:
    """
    Compute mask of high gradient regions in depth and normal maps.
    
    :param depth_map: Depth map tensor [H, W]
    :param normal_map: Optional normal map tensor [H, W, 3] or [3, H, W]
    :param depth_threshold: Threshold for depth gradient filtering
    :param normal_threshold: Threshold for normal gradient filtering
    :param edge_sigma: Gaussian smoothing sigma before gradient computation
    :return: Boolean mask tensor [H, W] where True indicates high-gradient regions to exclude
    """
    device = depth_map.device
    h, w = depth_map.shape[-2:]
    
    # Initialize edge mask
    edge_mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    
    # Depth gradient filtering
    if depth_map is not None:
        # Apply Gaussian smoothing
        depth_smooth = depth_map
        if edge_sigma > 0:
            # Create Gaussian kernel
            kernel_size = int(2 * edge_sigma * 3) + 1  # 3-sigma rule
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Simple Gaussian blur using conv2d
            sigma_tensor = torch.tensor(edge_sigma, device=device, dtype=depth_map.dtype)
            x = torch.arange(kernel_size, device=device, dtype=depth_map.dtype) - kernel_size // 2
            gaussian_1d = torch.exp(-0.5 * (x / sigma_tensor) ** 2)
            gaussian_1d = gaussian_1d / gaussian_1d.sum()
            
            # Apply separable Gaussian filter
            kernel_x = gaussian_1d.view(1, 1, 1, kernel_size)
            kernel_y = gaussian_1d.view(1, 1, kernel_size, 1)
            
            depth_input = depth_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            depth_smooth = torch.nn.functional.conv2d(depth_input, kernel_x, padding=(0, kernel_size//2))
            depth_smooth = torch.nn.functional.conv2d(depth_smooth, kernel_y, padding=(kernel_size//2, 0))
            depth_smooth = depth_smooth.squeeze(0).squeeze(0)  # [H, W]
        
        # Compute depth gradients using Sobel operators
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=depth_map.dtype, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=depth_map.dtype, device=device).view(1, 1, 3, 3)
        
        depth_input = depth_smooth.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        dx = torch.nn.functional.conv2d(depth_input, sobel_x, padding=1).squeeze()
        dy = torch.nn.functional.conv2d(depth_input, sobel_y, padding=1).squeeze()
        
        depth_grad_mag = torch.sqrt(dx**2 + dy**2)
        
        # Normalize by depth (relative gradient)
        relative_grad = depth_grad_mag / (depth_smooth + 1e-6)
        
        # Create depth edge mask
        depth_edge_mask = relative_grad > depth_threshold
        edge_mask |= depth_edge_mask
    
    # Normal gradient filtering
    if normal_map is not None:
        # Ensure normal_map is [H, W, 3]
        if normal_map.dim() == 3 and normal_map.shape[0] == 3:
            normal_map = normal_map.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        
        # Compute normal gradient magnitude for each component
        nx_grad_x = torch.gradient(normal_map[..., 0], dim=1)[0]
        nx_grad_y = torch.gradient(normal_map[..., 0], dim=0)[0]
        ny_grad_x = torch.gradient(normal_map[..., 1], dim=1)[0]
        ny_grad_y = torch.gradient(normal_map[..., 1], dim=0)[0]
        nz_grad_x = torch.gradient(normal_map[..., 2], dim=1)[0]
        nz_grad_y = torch.gradient(normal_map[..., 2], dim=0)[0]
        
        # Compute total normal gradient magnitude
        normal_grad_mag = torch.sqrt(
            nx_grad_x**2 + nx_grad_y**2 +
            ny_grad_x**2 + ny_grad_y**2 +
            nz_grad_x**2 + nz_grad_y**2
        )
        
        # Create normal edge mask
        normal_edge_mask = normal_grad_mag > normal_threshold
        edge_mask |= normal_edge_mask
    
    return edge_mask


def filter_gaussians_by_gradients(
    gaussians: dict[str, torch.Tensor],
    depth_map: torch.Tensor | None = None,
    normal_map: torch.Tensor | None = None,
    depth_threshold: float = 0.2,
    normal_threshold: float = 0.3,
    edge_sigma: float = 1.0,
    img_h: int | None = None,
    img_w: int | None = None,
) -> dict[str, torch.Tensor]:
    """
    Filter out Gaussians from high gradient regions in depth/normal maps.
    
    :param gaussians: Dictionary containing Gaussian parameters with 'xy', 'scale', 'rot', 'feat'
    :param depth_map: Optional depth map tensor [H, W]
    :param normal_map: Optional normal map tensor [H, W, 3] or [3, H, W]
    :param depth_threshold: Threshold for depth gradient filtering
    :param normal_threshold: Threshold for normal gradient filtering
    :param edge_sigma: Gaussian smoothing sigma before gradient computation
    :param img_h: Image height (required if depth_map or normal_map provided)
    :param img_w: Image width (required if depth_map or normal_map provided)
    :return: Filtered dictionary of Gaussian parameters
    """
    # If no depth or normal maps provided, return original gaussians
    if depth_map is None and normal_map is None:
        return gaussians
    
    # Get image dimensions
    if depth_map is not None:
        h, w = depth_map.shape[-2:]
    elif normal_map is not None:
        if normal_map.dim() == 3 and normal_map.shape[0] == 3:
            h, w = normal_map.shape[-2:]
        else:
            h, w = normal_map.shape[:2]
    else:
        if img_h is None or img_w is None:
            raise ValueError("img_h and img_w must be provided if depth_map and normal_map are None")
        h, w = img_h, img_w
    
    # Compute gradient mask
    edge_mask = compute_depth_normal_gradient_mask(
        depth_map if depth_map is not None else torch.zeros((h, w), device=gaussians["xy"].device),
        normal_map,
        depth_threshold,
        normal_threshold,
        edge_sigma,
    )
    
    # Convert Gaussian positions to pixel coordinates
    xy = gaussians["xy"]  # [N, 2] in [0, 1] range
    
    # Convert to pixel coordinates [0, w) x [0, h)
    pixel_x = (xy[:, 0] * w).long().clamp(0, w - 1)
    pixel_y = (xy[:, 1] * h).long().clamp(0, h - 1)
    
    # Check which Gaussians are in high-gradient regions
    gaussian_in_edge = edge_mask[pixel_y, pixel_x]
    
    # Keep only Gaussians NOT in high-gradient regions
    keep_mask = ~gaussian_in_edge
    
    if keep_mask.sum() == 0:
        # If all Gaussians would be filtered out, keep at least some
        print("Warning: All Gaussians would be filtered out by gradient filtering. Keeping 10% with lowest gradients.")
        # Keep 10% of Gaussians with lowest gradient values at their positions
        if depth_map is not None:
            gradient_values = compute_depth_normal_gradient_mask(
                depth_map, normal_map, depth_threshold * 10, normal_threshold * 10, edge_sigma
            ).float()
            gaussian_gradients = gradient_values[pixel_y, pixel_x]
            _, keep_indices = torch.topk(gaussian_gradients, k=max(1, len(gaussian_gradients) // 10), largest=False)
            keep_mask = torch.zeros_like(keep_mask)
            keep_mask[keep_indices] = True
        else:
            # Fallback: keep every 10th Gaussian
            keep_mask = torch.zeros_like(keep_mask)
            keep_mask[::10] = True
    
    # Filter all Gaussian parameters
    filtered_gaussians = {}
    for key, values in gaussians.items():
        filtered_gaussians[key] = values[keep_mask]
    
    num_original = len(gaussians["xy"])
    num_filtered = len(filtered_gaussians["xy"])
    print(f"Filtered {num_original - num_filtered} Gaussians from high-gradient regions ({num_filtered}/{num_original} remaining)")
    
    return filtered_gaussians


def compute_gradient_map(
    gt_images: torch.Tensor,
    gamma: float,
    img_h: int,
    img_w: int,
    save_path: str | Path | None = None,
) -> torch.Tensor:
    """
    Compute gradient-based probability map for Gaussian initialization (GPU-optimized).

    :param gt_images: Target images tensor [C, H, W]
    :param gamma: Gamma correction value
    :param img_h: Image height
    :param img_w: Image width
    :param save_path: Optional path to save gradient map visualization
    :return: Probability distribution based on gradient magnitude (GPU tensor)
    
    Usage example:
        # Load image
        images, _, _ = load_images("image.jpg", gamma=2.2)
        gt_images = torch.from_numpy(images).to("cuda")
        img_h, img_w = gt_images.shape[1], gt_images.shape[2]
        
        # Compute gradient-based probability map
        prob_map = compute_gradient_map(
            gt_images, gamma=2.2, img_h=img_h, img_w=img_w,
            save_path="gradient_map.png"
        )
        
        # Use for sampling positions
        pixel_xy = get_grid(img_h, img_w, device=gt_images.device).reshape(-1, 2)
        positions = sample_positions(pixel_xy, 1000, prob_map)
    """
    # Apply gamma correction on GPU
    gamma_corrected = torch.pow(gt_images, 1.0 / gamma)
    
    # Compute gradients on GPU
    g_norm = compute_image_gradients_gpu(gamma_corrected)
    
    # Normalize
    g_norm = g_norm / g_norm.max()

    if save_path:
        save_image(g_norm.cpu().numpy(), save_path)

    # Square and normalize to probability distribution
    g_norm_flat = g_norm.view(-1)
    g_norm_squared = torch.pow(g_norm_flat, 2.0)
    image_gradients = g_norm_squared / g_norm_squared.sum()
    
    return image_gradients


def sample_positions(
    pixel_xy: torch.Tensor,
    num_gaussians: int,
    prob: torch.Tensor | None = None,
    random_ratio: float = 0.3,
) -> torch.Tensor:
    """
    Sample Gaussian positions based on probability distribution with optional random sampling (GPU-optimized).

    :param pixel_xy: Grid of pixel coordinates [H*W, 2]
    :param num_gaussians: Number of Gaussians to initialize
    :param prob: Probability distribution for sampling (None for uniform, GPU tensor)
    :param random_ratio: Ratio of randomly placed Gaussians (0.0 to 1.0)
    :return: Sampled positions tensor [num_gaussians, 2]
    
    Usage example:
        # Create pixel grid
        pixel_xy = get_grid(h=512, w=512, device="cuda").reshape(-1, 2)
        
        # Sample with gradient-based probability
        prob_map = compute_gradient_map(gt_images, gamma=2.2, img_h=512, img_w=512)
        positions = sample_positions(pixel_xy, 1000, prob_map, random_ratio=0.3)
        
        # Sample purely random positions
        random_positions = sample_positions(pixel_xy, 500, prob=None, random_ratio=1.0)
    """
    num_pixels = pixel_xy.shape[0]
    num_random = round(random_ratio * num_gaussians)
    device = pixel_xy.device

    # Random sampling using torch.randperm for GPU efficiency
    if num_random > 0:
        random_indices = torch.randperm(num_pixels, device=device)[:num_random]
        positions_random = pixel_xy[random_indices]
    else:
        positions_random = torch.empty(0, 2, device=device, dtype=pixel_xy.dtype)

    # Probability-based sampling
    if prob is not None and num_gaussians > num_random:
        # Use torch.multinomial for GPU-based probability sampling
        remaining_count = num_gaussians - num_random
        prob_indices = torch.multinomial(prob, remaining_count, replacement=False)
        positions_other = pixel_xy[prob_indices]
        return torch.cat([positions_random, positions_other], dim=0)
    else:
        # All random if no probability distribution provided
        if num_gaussians > num_random:
            remaining_count = num_gaussians - num_random
            remaining_indices = torch.randperm(num_pixels, device=device)[:remaining_count]
            positions_remaining = pixel_xy[remaining_indices]
            return torch.cat([positions_random, positions_remaining], dim=0)
        return positions_random


def get_initial_features(
    gt_images: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """
    Sample initial RGB features from target image at Gaussian positions.

    :param gt_images: Target images tensor [C, H, W]
    :param positions: Gaussian positions [num_gaussians, 2] in [0, 1] range
    :return: Initial features tensor [num_gaussians, C]
    """
    with torch.no_grad():
        # gt_images [C, H, W]; positions [P, 2]
        # Convert positions to grid_sample format: [1, 1, P, 2] with range [-1, 1]
        grid_positions = positions[None, None, ...] * 2.0 - 1.0

        # Sample features using bilinear interpolation
        target_features = F.grid_sample(
            gt_images.unsqueeze(0), grid_positions, align_corners=False
        )

        # Reshape to [P, C]
        target_features = target_features[0, :, 0, :].permute(1, 0)

    return target_features


def initialize_gaussians(
    gt_images: torch.Tensor,
    num_gaussians: int,
    img_h: int,
    img_w: int,
    init_mode: str = "gradient",
    init_random_ratio: float = 0.3,
    init_scale: float = 5.0,
    gamma: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    disable_color_init: bool = False,
    log_dir: str | Path | None = None,
    depth_map: torch.Tensor | None = None,
    normal_map: torch.Tensor | None = None,
    filter_high_gradients: bool = True,
    depth_gradient_threshold: float = 0.2,
    normal_gradient_threshold: float = 0.3,
    gradient_edge_sigma: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    Initialize 2D Gaussians for image reconstruction.

    :param gt_images: Target images tensor [C, H, W]
    :param num_gaussians: Number of Gaussians to initialize
    :param img_h: Image height
    :param img_w: Image width
    :param init_mode: Initialization mode ("gradient" or "random")
    :param init_random_ratio: Ratio of randomly placed Gaussians
    :param init_scale: Initial Gaussian scale in pixels
    :param gamma: Gamma correction value
    :param device: Device to place tensors on
    :param dtype: Data type for tensors
    :param disable_color_init: If True, skip color initialization
    :param log_dir: Directory to save initialization visualizations
    :param depth_map: Optional depth map tensor [H, W] from MOGE for gradient filtering
    :param normal_map: Optional normal map tensor [H, W, 3] or [3, H, W] from MOGE for gradient filtering
    :param filter_high_gradients: Whether to filter out Gaussians from high-gradient regions
    :param depth_gradient_threshold: Threshold for depth gradient filtering (higher = more aggressive)
    :param normal_gradient_threshold: Threshold for normal gradient filtering (higher = more aggressive)
    :param gradient_edge_sigma: Gaussian smoothing sigma before gradient computation
    :return: Dictionary containing Gaussian parameters
    
    Usage example:
        # Load and prepare image
        images, _, _ = load_images("image.jpg", gamma=2.2)
        gt_images = torch.from_numpy(images).to("cuda")
        img_h, img_w = gt_images.shape[1], gt_images.shape[2]
        
        # Initialize with gradient-based placement and depth/normal filtering
        gaussians = initialize_gaussians(
            gt_images=gt_images,
            num_gaussians=10000,
            img_h=img_h,
            img_w=img_w,
            init_mode="gradient",
            init_random_ratio=0.3,
            init_scale=5.0,
            gamma=2.2,
            device="cuda",
            log_dir="logs/",
            depth_map=depth_tensor,  # From MOGE
            normal_map=normal_tensor,  # From MOGE
            filter_high_gradients=True,
            depth_gradient_threshold=0.2,
            normal_gradient_threshold=0.3
        )
        
        # Initialize without filtering
        gaussians_no_filter = initialize_gaussians(
            gt_images=gt_images,
            num_gaussians=5000,
            img_h=img_h,
            img_w=img_w,
            init_mode="random",
            filter_high_gradients=False
        )
        
        # Access Gaussian parameters
        positions = gaussians["xy"]      # [N, 2]
        scales = gaussians["scale"]      # [N, 2]  
        rotations = gaussians["rot"]     # [N, 1]
        features = gaussians["feat"]     # [N, C]
    """
    # Create pixel grid on the target device
    pixel_xy = get_grid(h=img_h, w=img_w, device=device, dtype=dtype).reshape(-1, 2)

    # Initialize positions based on mode
    if init_mode == "gradient":
        # Gradient-based initialization
        save_path = f"{log_dir}/gmap_res-{img_h}x{img_w}.png" if log_dir else None
        prob = compute_gradient_map(gt_images, gamma, img_h, img_w, save_path)
        xy = sample_positions(pixel_xy, num_gaussians, prob, init_random_ratio)
    else:
        # Random initialization - early exit without computing gradients
        xy = sample_positions(pixel_xy, num_gaussians, prob=None, random_ratio=1.0)

    # Initialize scale (uniform for all Gaussians)
    scale = torch.full((num_gaussians, 2), init_scale, dtype=dtype, device=device)

    # Initialize rotation (zero for all Gaussians)
    rot = torch.zeros((num_gaussians, 1), dtype=dtype, device=device)

    # Initialize features (sample colors from target image)
    if not disable_color_init:
        feat = get_initial_features(gt_images, xy).detach().clone()
    else:
        # Random initialization if color init is disabled
        num_channels = gt_images.shape[0]
        feat = torch.rand((num_gaussians, num_channels), dtype=dtype, device=device)

    # Create initial Gaussian dictionary
    gaussians = {"xy": xy, "scale": scale, "rot": rot, "feat": feat}
    
    # Apply gradient-based filtering if requested and maps are provided
    if filter_high_gradients and (depth_map is not None or normal_map is not None):
        gaussians = filter_gaussians_by_gradients(
            gaussians=gaussians,
            depth_map=depth_map,
            normal_map=normal_map,
            depth_threshold=depth_gradient_threshold,
            normal_threshold=normal_gradient_threshold,
            edge_sigma=gradient_edge_sigma,
            img_h=img_h,
            img_w=img_w,
        )

    return gaussians


def initialize_from_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """
    Load Gaussian parameters from a checkpoint file.

    :param checkpoint_path: Path to checkpoint file
    :return: Dictionary containing Gaussian parameters and optimization state
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    return checkpoint


@dataclass
class InitializerConfig:
    """Configuration for Gaussian initialization."""

    input_path: str = "images/anime-1_2k.png"
    """Path to input image"""

    num_gaussians: int = 10000
    """Number of Gaussians"""

    init_mode: str = "gradient"
    """Initialization mode: gradient or random"""

    init_random_ratio: float = 0.3
    """Ratio of randomly placed Gaussians"""

    init_scale: float = 5.0
    """Initial Gaussian scale in pixels"""

    gamma: float = 1.0
    """Gamma correction value"""
    
    filter_high_gradients: bool = True
    """Whether to filter out Gaussians from high-gradient regions"""
    
    depth_gradient_threshold: float = 0.2
    """Threshold for depth gradient filtering (higher = more aggressive)"""
    
    normal_gradient_threshold: float = 0.3
    """Threshold for normal gradient filtering (higher = more aggressive)"""
    
    gradient_edge_sigma: float = 1.0
    """Gaussian smoothing sigma before gradient computation"""


def main(config: InitializerConfig) -> None:
    """Main function for Gaussian initialization.

    :param config: Configuration for initialization
    """
    # Load image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, channels, _ = load_images(f"media/{config.input_path}", gamma=config.gamma)
    gt_images = torch.from_numpy(images).to(device)
    img_h, img_w = gt_images.shape[1], gt_images.shape[2]

    # Initialize Gaussians
    gaussians = initialize_gaussians(
        gt_images=gt_images,
        num_gaussians=config.num_gaussians,
        img_h=img_h,
        img_w=img_w,
        init_mode=config.init_mode,
        init_random_ratio=config.init_random_ratio,
        init_scale=config.init_scale,
        gamma=config.gamma,
        device=device,
    )

    print(f"Initialized {config.num_gaussians} Gaussians using {config.init_mode} mode")
    print(f"Position shape: {gaussians['xy'].shape}")
    print(f"Scale shape: {gaussians['scale'].shape}")
    print(f"Rotation shape: {gaussians['rot'].shape}")
    print(f"Feature shape: {gaussians['feat'].shape}")


if __name__ == "__main__":
    tyro.cli(main)
