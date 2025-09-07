"""
Utility functions for DepthDensifier.
"""

from pathlib import Path
import pycolmap
import numpy as np


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