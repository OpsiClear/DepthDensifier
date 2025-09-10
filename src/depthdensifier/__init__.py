"""DepthDensifier: Densify COLMAP point clouds using depth maps."""

from .depth_refiner import DepthRefiner, RefinerConfig
from .point_sampler import sample_points, visualize_points

__version__ = "0.1.0"
__all__ = [
    "DepthRefiner", 
    "RefinerConfig",
    "sample_points",
    "visualize_points",
]
