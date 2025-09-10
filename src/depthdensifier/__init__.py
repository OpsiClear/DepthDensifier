"""DepthDensifier: Densify COLMAP point clouds using depth maps."""

from .depth_refiner import DepthRefiner, RefinerConfig
from .point_sampler import sample_points, visualize_points
from .utils import (
    load_colmap_model,
    unproject_points,
    find_colmap_datasets,
    validate_colmap_reconstruction,
)

__version__ = "0.1.0"
__all__ = [
    "DepthRefiner", 
    "RefinerConfig",
    "sample_points",
    "visualize_points",
    "load_colmap_model",
    "unproject_points",
    "find_colmap_datasets",
    "validate_colmap_reconstruction",
]
