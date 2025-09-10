"""
Utility functions for DepthDensifier.
"""

from pathlib import Path
from typing import List, Set
import numpy as np
import pycolmap
import logging

logger = logging.getLogger(__name__)


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


def unproject_points(points2D: np.ndarray, depth: np.ndarray, camera: pycolmap.Camera) -> np.ndarray:
    """
    Unproject 2D image points to 3D camera coordinates using depth values.
    
    Args:
        points2D: Array of 2D points in pixel coordinates, shape (N, 2)
        depth: Array of depth values for each point, shape (N,)
        camera: COLMAP camera model with intrinsic parameters
        
    Returns:
        Array of 3D points in camera coordinates, shape (N, 3)
        
    Example:
        ```python
        # Get 2D points and their depths
        points2D = np.array([[320, 240], [640, 480]])
        depths = np.array([2.5, 3.0])
        
        # Unproject to 3D
        points3D_camera = unproject_points(points2D, depths, camera)
        
        # Transform to world coordinates
        points3D_world = image.cam_from_world().inverse() * points3D_camera
        ```
    """
    fx, fy, cx, cy = camera.params
    u, v = points2D[:, 0], points2D[:, 1]
    x_normalized = (u - cx) / fx
    y_normalized = (v - cy) / fy
    points3D_camera = np.stack(
        [x_normalized * depth, y_normalized * depth, depth], axis=-1
    )
    return points3D_camera


def find_colmap_datasets(
    data_dir: Path, 
    skip_items: Set[str] | None = None,
    check_colmap_files: bool = True
) -> List[str]:
    """
    Find all valid COLMAP datasets in a directory.
    
    Args:
        data_dir: Root directory to search for datasets
        skip_items: Set of directory names to skip (default: common non-dataset folders)
        check_colmap_files: Whether to verify COLMAP binary files exist
        
    Returns:
        Sorted list of dataset names (directory names) that contain valid COLMAP reconstructions
        
    Example:
        ```python
        # Find all datasets in data directory
        datasets = find_colmap_datasets(Path("data/360_v2"))
        print(f"Found {len(datasets)} datasets: {', '.join(datasets)}")
        
        # Process each dataset
        for dataset_name in datasets:
            recon_path = data_dir / dataset_name / "sparse" / "0"
            images_path = data_dir / dataset_name / "images"
            # ... process dataset
        ```
    """
    if skip_items is None:
        skip_items = {'flowers.txt', 'treehill.txt', 'outputs', 'temp', '__pycache__'}
    
    datasets = []
    
    if not data_dir.exists():
        logger.warning(f"Data directory {data_dir} does not exist")
        return datasets
    
    for item in data_dir.iterdir():
        if item.is_dir() and item.name not in skip_items:
            # Check if it has the required structure
            sparse_dir = item / "sparse" / "0"
            images_dir = item / "images"
            
            if sparse_dir.exists() and images_dir.exists():
                if check_colmap_files:
                    # Check for COLMAP binary files
                    cameras_bin = sparse_dir / "cameras.bin"
                    images_bin = sparse_dir / "images.bin"
                    points3d_bin = sparse_dir / "points3D.bin"
                    
                    if cameras_bin.exists() and images_bin.exists() and points3d_bin.exists():
                        datasets.append(item.name)
                        logger.info(f"Found valid dataset: {item.name}")
                    else:
                        logger.warning(f"Dataset {item.name} missing COLMAP files, skipping")
                else:
                    # Just check directory structure
                    datasets.append(item.name)
                    logger.info(f"Found dataset: {item.name}")
            else:
                logger.debug(f"Dataset {item.name} missing sparse/0 or images directory, skipping")
    
    return sorted(datasets)


def validate_colmap_reconstruction(recon_path: Path) -> bool:
    """
    Validate that a COLMAP reconstruction is complete and valid.
    
    Args:
        recon_path: Path to COLMAP reconstruction directory (e.g., "sparse/0")
        
    Returns:
        True if reconstruction is valid, False otherwise
        
    Example:
        ```python
        recon_path = Path("data/bicycle/sparse/0")
        if validate_colmap_reconstruction(recon_path):
            rec = pycolmap.Reconstruction(recon_path)
            # Process reconstruction
        else:
            print("Invalid reconstruction")
        ```
    """
    if not recon_path.exists():
        return False
    
    # Check for required binary files
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    for filename in required_files:
        if not (recon_path / filename).exists():
            return False
    
    try:
        # Try to load the reconstruction
        rec = pycolmap.Reconstruction(recon_path)
        # Check if it has registered images and points
        if rec.num_reg_images() == 0 or rec.num_points3D() == 0:
            return False
        return True
    except Exception as e:
        logger.warning(f"Failed to load reconstruction from {recon_path}: {e}")
        return False