"""
Simple and Robust 2D Point Sampling Module

GPU-accelerated point sampling from images using gradient-based and random methods.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import TypedDict, Literal, TypeAlias
import numpy.typing as npt

# Type aliases for clarity
ImageArray: TypeAlias = npt.NDArray[np.float32]
TensorType: TypeAlias = torch.Tensor
DeviceType: TypeAlias = str | torch.device | None
ImageInput: TypeAlias = ImageArray | TensorType | str | Path | Image.Image
SamplingStrategy: TypeAlias = Literal["random", "edges", "mixed"]


class SampleResult(TypedDict):
    """Type definition for sampling result dictionary."""
    positions: ImageArray  # (N, 2) normalized coordinates [0, 1]
    pixels: ImageArray     # (N, 2) integer pixel coordinates
    weights: ImageArray    # (N,) importance weights
    colors: ImageArray     # (N, 3) RGB colors (optional)


def ensure_tensor(
    image: ImageInput,
    device: DeviceType = None,
    dtype: torch.dtype = torch.float32
) -> TensorType:
    """Convert various input types to a torch tensor robustly.
    
    Args:
        image: Input image in various formats
        device: Target device (None for auto-detection)
        dtype: Target data type
        
    Returns:
        Image tensor [C, H, W] in range [0, 1]
    """
    # Auto-detect device if not specified
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Handle different input types
    if isinstance(image, (str, Path)):
        # Load from file
        try:
            pil_image = Image.open(image).convert("RGB")
            image = np.array(pil_image, dtype=np.float32) / 255.0
        except Exception as e:
            raise ValueError(f"Failed to load image from {image}: {e}")
    
    elif isinstance(image, Image.Image):
        # PIL Image
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = np.array(image, dtype=np.float32) / 255.0
    
    elif isinstance(image, np.ndarray):
        # NumPy array
        image = image.astype(np.float32)
        
        # Normalize if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        # Handle grayscale
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Ensure HWC format
        if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
            if image.shape[0] < image.shape[2]:  # Likely CHW
                image = np.transpose(image, (1, 2, 0))
    
    elif isinstance(image, torch.Tensor):
        # Already a tensor - just ensure correct format
        if image.device != device:
            image = image.to(device)
        if image.dtype != dtype:
            image = image.to(dtype)
        
        # Ensure [0, 1] range
        if image.max() > 1.0:
            image = image / 255.0
        
        # Ensure CHW format
        if image.ndim == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        elif image.ndim == 3 and image.shape[2] in [1, 3, 4]:
            image = image.permute(2, 0, 1)
        
        # Ensure 3 channels
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3]  # Drop alpha
        
        return image.contiguous()
    
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    # Convert numpy to tensor
    if isinstance(image, np.ndarray):
        # Ensure HWC -> CHW
        if image.ndim == 3 and image.shape[2] in [1, 3, 4]:
            image = np.transpose(image, (2, 0, 1))
        
        # Create tensor
        image = torch.from_numpy(image).to(device=device, dtype=dtype)
    
    # Final validation
    if image.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {image.shape}")
    
    if image.shape[0] not in [1, 3]:
        if image.shape[0] == 4:
            image = image[:3]  # Drop alpha channel
        else:
            raise ValueError(f"Expected 1 or 3 channels, got {image.shape[0]}")
    
    # Ensure 3 channels for consistency
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    
    return image.contiguous()


@torch.no_grad()
def compute_edge_map(
    image: TensorType,
    method: Literal["sobel", "simple"] = "sobel",
    blur_size: int | None = None
) -> TensorType:
    """Compute edge/gradient map from image.
    
    Args:
        image: Image tensor [C, H, W]
        method: Edge detection method ('sobel' or 'simple')
        blur_size: Optional Gaussian blur kernel size (must be odd)
        
    Returns:
        Edge map [H, W] with values in [0, 1]
    """
    device = image.device
    dtype = image.dtype
    
    # Optional blur for noise reduction
    if blur_size is not None and blur_size > 1:
        # Ensure odd kernel size
        if blur_size % 2 == 0:
            blur_size += 1
        
        # Apply Gaussian blur
        sigma = blur_size / 3.0
        image = F.gaussian_blur(image.unsqueeze(0), kernel_size=blur_size).squeeze(0)
    
    if method == "sobel":
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=dtype, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=dtype, device=device).view(1, 1, 3, 3)
        
        # Apply to all channels at once
        C = image.shape[0]
        sobel_x = sobel_x.repeat(C, 1, 1, 1)
        sobel_y = sobel_y.repeat(C, 1, 1, 1)
        
        image_batch = image.unsqueeze(0)
        gx = F.conv2d(image_batch, sobel_x, padding=1, groups=C)
        gy = F.conv2d(image_batch, sobel_y, padding=1, groups=C)
        
        # Magnitude across all channels
        edge_map = torch.sqrt((gx**2 + gy**2).sum(dim=1)).squeeze(0)
        
    else:  # simple gradient
        # Simple gradient using finite differences
        gx = image[:, :, 1:] - image[:, :, :-1]
        gy = image[:, 1:, :] - image[:, :-1, :]
        
        # Pad to original size
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        
        # Magnitude
        edge_map = torch.sqrt((gx**2 + gy**2).sum(dim=0))
    
    # Normalize to [0, 1]
    edge_min = edge_map.min()
    edge_max = edge_map.max()
    
    if edge_max > edge_min:
        edge_map = (edge_map - edge_min) / (edge_max - edge_min)
    else:
        edge_map = torch.zeros_like(edge_map)
    
    return edge_map


@torch.no_grad()
def sample_points(
    image: ImageInput,
    num_points: int = 1000,
    strategy: SamplingStrategy = "mixed",
    edge_weight: float = 0.7,
    device: DeviceType = None,
    return_colors: bool = True,
    seed: int | None = None
) -> SampleResult:
    """Sample points from an image with various strategies.
    
    Simple API for point sampling with automatic format handling.
    
    Args:
        image: Input image (file path, PIL Image, numpy array, or torch tensor)
        num_points: Number of points to sample
        strategy: Sampling strategy:
            - 'random': Uniform random sampling
            - 'edges': Sample from high-gradient regions
            - 'mixed': Combination of random and edge-based (default)
        edge_weight: Weight for edge-based sampling in mixed mode (0-1)
        device: Device for computation (None for auto-detection)
        return_colors: Whether to return colors at sampled points
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
            - 'positions': (N, 2) normalized coordinates in [0, 1]
            - 'pixels': (N, 2) integer pixel coordinates
            - 'colors': (N, 3) RGB colors at points (if requested)
            - 'weights': (N,) sampling weights/importance
        
    Example:
        ```python
        # From file
        points = sample_points("image.jpg", num_points=1000)
        
        # From numpy array
        img_array = np.random.rand(256, 256, 3)
        points = sample_points(img_array, strategy="edges")
        
        # From torch tensor
        img_tensor = torch.rand(3, 512, 512).cuda()
        points = sample_points(img_tensor, strategy="mixed", edge_weight=0.8)
        ```
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Validate inputs
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")
    
    strategies: list[SamplingStrategy] = ["random", "edges", "mixed"]
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Must be one of {strategies}")
    
    edge_weight = np.clip(edge_weight, 0.0, 1.0)
    
    # Convert input to tensor
    try:
        image_tensor = ensure_tensor(image, device=device)
    except Exception as e:
        raise RuntimeError(f"Failed to process image: {e}")
    
    C, H, W = image_tensor.shape
    total_pixels = H * W
    
    # Ensure we don't sample more points than pixels
    num_points = min(num_points, total_pixels)
    
    # Determine sampling strategy
    if strategy == "random":
        num_edge_points = 0
        num_random_points = num_points
    elif strategy == "edges":
        num_edge_points = num_points
        num_random_points = 0
    else:  # mixed
        num_edge_points = int(num_points * edge_weight)
        num_random_points = num_points - num_edge_points
    
    # Collect sampled indices
    sampled_indices: list[TensorType] = []
    
    # Random sampling
    if num_random_points > 0:
        random_indices = torch.randperm(total_pixels, device=image_tensor.device)[:num_random_points]
        sampled_indices.append(random_indices)
    
    # Edge-based sampling
    edge_map: TensorType | None = None
    if num_edge_points > 0:
        # Compute edge map
        edge_map = compute_edge_map(image_tensor, method="sobel")
        
        # Convert to probability distribution
        edge_probs = edge_map.view(-1)
        edge_probs = edge_probs ** 2  # Square to emphasize strong edges
        
        # Handle edge case of uniform image
        if edge_probs.sum() > 0:
            edge_probs = edge_probs / edge_probs.sum()
        else:
            # Fallback to uniform if no edges detected
            edge_probs = torch.ones_like(edge_probs) / len(edge_probs)
        
        # Sample based on edge probability
        try:
            edge_indices = torch.multinomial(edge_probs, num_edge_points, replacement=False)
            sampled_indices.append(edge_indices)
        except RuntimeError:
            # Fallback if multinomial fails (e.g., too many samples requested)
            fallback_indices = torch.randperm(total_pixels, device=image_tensor.device)[:num_edge_points]
            sampled_indices.append(fallback_indices)
    
    # Combine all indices
    if sampled_indices:
        all_indices = torch.cat(sampled_indices)
    else:
        # Shouldn't happen, but handle gracefully
        all_indices = torch.randperm(total_pixels, device=image_tensor.device)[:num_points]
    
    # Convert indices to coordinates
    y_coords = all_indices // W
    x_coords = all_indices % W
    
    # Create output dictionary
    result: dict[str, ImageArray] = {}
    
    # Pixel coordinates
    pixel_coords = torch.stack([x_coords, y_coords], dim=1)
    result['pixels'] = pixel_coords.cpu().numpy()
    
    # Normalized positions [0, 1]
    positions = pixel_coords.float()
    positions[:, 0] /= max(W - 1, 1)
    positions[:, 1] /= max(H - 1, 1)
    result['positions'] = positions.cpu().numpy()
    
    # Sample colors if requested
    if return_colors:
        # Efficient gathering using flattened indexing
        image_flat = image_tensor.view(C, -1)
        colors = image_flat[:, all_indices].T  # (N, C)
        result['colors'] = colors.cpu().numpy()
    
    # Compute importance weights (based on edge strength)
    if num_edge_points > 0 and edge_map is not None:
        edge_flat = edge_map.view(-1)
        weights = edge_flat[all_indices]
    else:
        weights = torch.ones(num_points, device=image_tensor.device)
    
    # Normalize weights
    weights = weights / weights.max() if weights.max() > 0 else weights
    result['weights'] = weights.cpu().numpy()
    
    return result  # type: ignore


def visualize_points(
    image: ImageInput,
    points: ImageArray | TensorType | SampleResult,
    point_size: int = 3,
    point_color: tuple[int, int, int] = (255, 0, 0),
    save_path: str | Path | None = None,
    show_weights: bool = False
) -> ImageArray:
    """Visualize sampled points on an image.
    
    Args:
        image: Input image
        points: Point positions as array or result dict from sample_points
        point_size: Size of points to draw
        point_color: RGB color for points (ignored if show_weights=True)
        save_path: Optional path to save visualization
        show_weights: Color points by their weights (if available)
        
    Returns:
        Visualization as numpy array [H, W, 3]
    """
    # Handle input image
    image_tensor = ensure_tensor(image, device="cpu")
    img_np = image_tensor.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    
    # Get image dimensions
    H, W = img_np.shape[:2]
    
    # Handle points input
    positions: ImageArray
    weights: ImageArray | None = None
    
    if isinstance(points, dict):
        # Result from sample_points
        positions = points.get('positions', points.get('pixels'))  # type: ignore
        weights = points.get('weights', None) if show_weights else None  # type: ignore
    else:
        positions = points if isinstance(points, np.ndarray) else points.cpu().numpy()
        weights = None
    
    # Create visualization
    img_vis = (img_np * 255).astype(np.uint8)
    if img_vis.shape[2] == 1:
        img_vis = np.repeat(img_vis, 3, axis=2)
    
    img_pil = Image.fromarray(img_vis)
    
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img_pil)
    
    # Draw points
    for i, point in enumerate(positions):
        # Handle both normalized [0,1] and pixel coordinates
        if positions.max() <= 1.0:
            # Normalized coordinates
            x = int(point[0] * (W - 1))
            y = int(point[1] * (H - 1))
        else:
            # Pixel coordinates
            x = int(point[0])
            y = int(point[1])
        
        # Determine color
        color: tuple[int, int, int]
        if weights is not None and i < len(weights):
            # Color by weight (red = high, blue = low)
            weight = float(weights[i])
            color = (
                int(255 * weight),
                int(128 * (1 - abs(weight - 0.5) * 2)),
                int(255 * (1 - weight))
            )
        else:
            color = point_color
        
        # Draw point
        draw.ellipse(
            [x - point_size, y - point_size, x + point_size, y + point_size],
            fill=color
        )
    
    # Convert back to numpy
    img_vis = np.array(img_pil)
    
    # Save if requested
    if save_path:
        img_pil.save(save_path)
        print(f"Saved visualization to {save_path}")
    
    return img_vis


# Simple example usage
if __name__ == "__main__":
    print("Testing point sampler with Python 3.12+ type hints...")
    
    # Create test image with pattern
    H, W = 256, 256
    x = np.linspace(0, 4*np.pi, W)
    y = np.linspace(0, 4*np.pi, H)
    xx, yy = np.meshgrid(x, y)
    
    # Create interesting pattern
    img = np.stack([
        np.sin(xx) * np.cos(yy),
        np.cos(xx*0.5) * np.sin(yy*0.5),
        np.ones_like(xx) * 0.5
    ], axis=2).astype(np.float32)
    
    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min())
    
    print(f"Test image shape: {img.shape}")
    
    # Test different strategies
    strategies: list[SamplingStrategy] = ["random", "edges", "mixed"]
    
    for strategy in strategies:
        print(f"\nTesting {strategy} sampling...")
        result = sample_points(
            img,
            num_points=500,
            strategy=strategy,
            edge_weight=0.7,
            return_colors=True
        )
        
        print(f"  Sampled {len(result['positions'])} points")
        print(f"  Position range: [{result['positions'].min():.3f}, {result['positions'].max():.3f}]")
        print(f"  Weight range: [{result['weights'].min():.3f}, {result['weights'].max():.3f}]")
        
        # Visualize
        vis = visualize_points(
            img, 
            result,
            point_size=2,
            save_path=f"sample_{strategy}.png",
            show_weights=True
        )
    
    print("\nTest completed successfully!")