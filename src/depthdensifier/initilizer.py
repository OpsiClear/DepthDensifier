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
    :return: Dictionary containing Gaussian parameters
    
    Usage example:
        # Load and prepare image
        images, _, _ = load_images("image.jpg", gamma=2.2)
        gt_images = torch.from_numpy(images).to("cuda")
        img_h, img_w = gt_images.shape[1], gt_images.shape[2]
        
        # Initialize with gradient-based placement
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
            log_dir="logs/"
        )
        
        # Initialize with random placement
        gaussians_random = initialize_gaussians(
            gt_images=gt_images,
            num_gaussians=5000,
            img_h=img_h,
            img_w=img_w,
            init_mode="random",
            disable_color_init=False
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

    return {"xy": xy, "scale": scale, "rot": rot, "feat": feat}


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
