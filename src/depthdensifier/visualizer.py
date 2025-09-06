"""
Interactive 3D visualizer for COLMAP reconstructions and RGBD data from MoGe.

This module provides an interactive Plotly-based visualization tool with comprehensive
controls for comparing depth estimation results and sparse reconstructions:

Features:
- COLMAP sparse/dense point clouds with camera poses
- RGBD point clouds from MoGe depth estimation
- 2D-3D correspondence visualization for specific images
- Normal map visualizations in world space
- Side-by-side comparison views
- Unified left-panel control interface with:
  * Main viewing presets (Sparse+Corresp, Initial MoGe, Refined MoGe, etc.)
  * Show/Hide all controls
  * View controls (Orthographic/Perspective, Reset View)
  * Camera visibility toggles

Usage:
    from depthdensifier.visualizer import COLMAPVisualizer
    
    # Create visualizer
    viz = COLMAPVisualizer()
    
    # Load COLMAP reconstruction
    viz.load_colmap("path/to/colmap/sparse/0")
    
    # Add initial depth estimation
    viz.add_rgbd_pointcloud(
        depth_map=initial_depth,
        rgb_image=image,
        K=intrinsics,
        cam_from_world=extrinsics,
        name="Initial MoGe RGBD"
    )
    
    # Add refined depth estimation
    viz.add_rgbd_pointcloud(
        depth_map=refined_depth,
        rgb_image=image,
        K=intrinsics,
        cam_from_world=extrinsics,
        name="Refined RGBD"
    )
    
    # Add correspondence visualization for specific image
    viz.add_image_correspondences(image_id=1)
    
    # Create interactive visualization with unified controls
    fig = viz.create_figure()
    fig.show()
    
    # Or save as HTML
    viz.save_html("visualization.html")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import pycolmap
except ImportError:
    pycolmap = None


@dataclass
class PointCloud:
    """Container for point cloud data."""
    points: np.ndarray  # (N, 3) array of 3D points
    colors: Optional[np.ndarray] = None  # (N, 3) array of RGB colors [0-255]
    normals: Optional[np.ndarray] = None  # (N, 3) array of normal vectors
    name: str = "Point Cloud"
    visible: bool = True
    point_size: int = 1
    opacity: float = 0.8


@dataclass
class Camera:
    """Container for camera data."""
    position: np.ndarray  # (3,) camera center in world coordinates
    orientation: Optional[np.ndarray] = None  # (3, 3) rotation matrix
    name: str = "Camera"
    color: str = "cyan"
    size: int = 8


@dataclass
class COLMAPVisualizer:
    """Interactive visualizer for COLMAP and RGBD data."""
    
    point_clouds: list[PointCloud] = field(default_factory=list)
    cameras: list[Camera] = field(default_factory=list)
    colmap_reconstruction: Optional[object] = None
    max_points_display: int = 100000
    image_correspondences: Optional[dict] = None  # Store 2D-3D correspondences for current image
    
    def load_colmap(self, path: Union[str, Path]) -> None:
        """Load COLMAP reconstruction from directory.
        
        :param path: Path to COLMAP reconstruction (e.g., sparse/0)
        """
        if pycolmap is None:
            raise ImportError("pycolmap is required for loading COLMAP models. Install with: pip install pycolmap")
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"COLMAP path not found: {path}")
        
        # Load reconstruction
        self.colmap_reconstruction = pycolmap.Reconstruction(str(path))
        
        # Extract sparse points
        points = []
        colors = []
        for point3D in self.colmap_reconstruction.points3D.values():
            points.append(point3D.xyz)
            colors.append(point3D.color)
        
        if points:
            points = np.array(points)
            colors = np.array(colors)
            
            self.add_pointcloud(
                points=points,
                colors=colors,
                name="COLMAP Sparse",
                point_size=1,
                opacity=0.7
            )
        
        # Extract camera poses
        for img_id, image in self.colmap_reconstruction.images.items():
            camera_center = image.projection_center()
            self.add_camera(
                position=camera_center,
                name=f"Cam_{img_id}",
                color="lightblue",
                size=6
            )
    
    def add_pointcloud(self, points: np.ndarray, 
                      colors: Optional[np.ndarray] = None,
                      normals: Optional[np.ndarray] = None,
                      name: str = "Point Cloud",
                      visible: bool = True,
                      point_size: int = 1,
                      opacity: float = 0.8) -> None:
        """Add a point cloud to the visualization.
        
        :param points: (N, 3) array of 3D points
        :param colors: Optional (N, 3) array of RGB colors [0-255]
        :param normals: Optional (N, 3) array of normal vectors
        :param name: Name for the point cloud
        :param visible: Initial visibility
        :param point_size: Size of points in pixels
        :param opacity: Opacity of points [0-1]
        """
        pc = PointCloud(
            points=points,
            colors=colors,
            normals=normals,
            name=name,
            visible=visible,
            point_size=point_size,
            opacity=opacity
        )
        self.point_clouds.append(pc)
    
    def add_camera(self, position: np.ndarray,
                  orientation: Optional[np.ndarray] = None,
                  name: str = "Camera",
                  color: str = "cyan",
                  size: int = 8) -> None:
        """Add a camera to the visualization.
        
        :param position: Camera center in world coordinates
        :param orientation: Optional (3, 3) rotation matrix
        :param name: Name for the camera
        :param color: Color for camera marker
        :param size: Size of camera marker
        """
        cam = Camera(
            position=position,
            orientation=orientation,
            name=name,
            color=color,
            size=size
        )
        self.cameras.append(cam)
    
    def add_image_correspondences(self, image_id: int) -> None:
        """Add 2D-3D correspondences for a specific COLMAP image.
        
        :param image_id: COLMAP image ID to extract correspondences from
        """
        if self.colmap_reconstruction is None:
            raise ValueError("COLMAP reconstruction must be loaded first")
        
        if image_id not in self.colmap_reconstruction.images:
            raise ValueError(f"Image ID {image_id} not found in reconstruction")
        
        image = self.colmap_reconstruction.images[image_id]
        
        # Extract visible 3D points for this image
        points_3d = []
        points_2d = []
        colors = []
        
        for point2D in image.points2D:
            if point2D.has_point3D():
                point3D_id = point2D.point3D_id
                if point3D_id in self.colmap_reconstruction.points3D:
                    point3D = self.colmap_reconstruction.points3D[point3D_id]
                    points_3d.append(point3D.xyz)
                    points_2d.append(point2D.xy)
                    colors.append(point3D.color)
        
        if points_3d:
            points_3d = np.array(points_3d)
            colors = np.array(colors)
            
            # Add as a special point cloud in red
            self.add_pointcloud(
                points=points_3d,
                colors=None,  # Use red color instead of original
                name=f"Image {image_id} Correspondences",
                visible=False,  # Initially hidden
                point_size=2,
                opacity=0.9
            )
            
            # Store correspondence info
            self.image_correspondences = {
                'image_id': image_id,
                'points_2d': np.array(points_2d),
                'points_3d': points_3d,
                'image_name': image.name
            }
    
    def add_rgbd_pointcloud(self, depth_map: np.ndarray,
                           rgb_image: Optional[np.ndarray] = None,
                           K: np.ndarray = None,
                           cam_from_world: np.ndarray = None,
                           mask: Optional[np.ndarray] = None,
                           normal_map: Optional[np.ndarray] = None,
                           name: str = "RGBD Point Cloud",
                           **kwargs) -> np.ndarray:
        """Add a point cloud from RGBD data.
        
        :param depth_map: (H, W) depth map
        :param rgb_image: Optional (H, W, 3) RGB image [0-255]
        :param K: (3, 3) camera intrinsic matrix
        :param cam_from_world: (3, 4) or (4, 4) camera extrinsic matrix
        :param mask: Optional (H, W) validity mask
        :param normal_map: Optional (H, W, 3) normal map
        :param name: Name for the point cloud
        :param kwargs: Additional arguments for add_pointcloud
        :return: (N, 3) array of 3D points in world space
        """
        if K is None or cam_from_world is None:
            raise ValueError("Camera intrinsics (K) and extrinsics (cam_from_world) are required")
        
        # Convert depth to point cloud
        points_world, colors = self._depth_to_pointcloud(
            depth_map, K, cam_from_world, rgb_image, mask
        )
        
        # Process normals if provided
        normals_world = None
        if normal_map is not None and mask is not None:
            # Transform normals from camera to world space
            normals_world = self._transform_normals(normal_map, cam_from_world, mask)
        
        # Add to visualization
        self.add_pointcloud(
            points=points_world,
            colors=colors,
            normals=normals_world,
            name=name,
            **kwargs
        )
        
        return points_world
    
    def _depth_to_pointcloud(self, depth: np.ndarray, K: np.ndarray, 
                            cam_from_world: np.ndarray,
                            color: Optional[np.ndarray] = None,
                            mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert depth map to 3D point cloud in world coordinates.
        
        :param depth: (H, W) depth map
        :param K: (3, 3) camera intrinsic matrix
        :param cam_from_world: (3, 4) or (4, 4) camera extrinsic matrix
        :param color: Optional (H, W, 3) RGB image
        :param mask: Optional (H, W) validity mask
        :return: Points (N, 3) and optionally colors (N, 3)
        """
        h, w = depth.shape
        
        # Create pixel grid (using xy indexing for correct image coordinates)
        xx, yy = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
        pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1).reshape(-1, 3)
        
        # Apply mask if provided
        if mask is not None:
            valid = mask.flatten() > 0
        else:
            valid = depth.flatten() > 0
        
        pixels = pixels[valid]
        depths = depth.flatten()[valid]
        
        # Unproject to camera space
        K_inv = np.linalg.inv(K)
        rays = (K_inv @ pixels.T).T
        points_cam = rays * depths[:, np.newaxis]
        
        # Transform to world space
        if cam_from_world.shape[0] == 3:
            # 3x4 matrix - convert to 4x4
            cam_from_world = np.vstack([cam_from_world, [0, 0, 0, 1]])
        
        # Invert to get world_from_cam
        world_from_cam = np.linalg.inv(cam_from_world)
        
        # Transform points
        points_cam_h = np.hstack([points_cam, np.ones((len(points_cam), 1))])
        points_world = (world_from_cam @ points_cam_h.T).T[:, :3]
        
        # Get colors if provided
        colors = None
        if color is not None:
            colors = color.reshape(-1, 3)[valid]
            # Ensure colors are in [0, 255] range
            if colors.size > 0 and np.max(colors) <= 1.0:
                colors = (colors * 255).astype(np.uint8)
        
        return points_world, colors
    
    def _transform_normals(self, normal_map: np.ndarray, 
                          cam_from_world: np.ndarray,
                          mask: np.ndarray) -> np.ndarray:
        """Transform normal vectors from camera to world space.
        
        :param normal_map: (H, W, 3) normal map in camera space
        :param cam_from_world: (3, 4) or (4, 4) camera extrinsic matrix
        :param mask: (H, W) validity mask
        :return: (N, 3) normal vectors in world space
        """
        # Get rotation matrix from cam_from_world
        if cam_from_world.shape[0] == 3:
            R_cam_from_world = cam_from_world[:, :3]
        else:
            R_cam_from_world = cam_from_world[:3, :3]
        
        # Invert rotation to get world_from_cam
        R_world_from_cam = R_cam_from_world.T
        
        # Flatten and filter normals
        valid = mask.flatten() > 0
        normals_cam = normal_map.reshape(-1, 3)[valid]
        
        # Transform to world space
        normals_world = (R_world_from_cam @ normals_cam.T).T
        
        # Normalize
        norms = np.linalg.norm(normals_world, axis=1, keepdims=True)
        normals_world = normals_world / (norms + 1e-8)
        
        return normals_world
    
    def create_figure(self, 
                     title: str = "3D Point Cloud Visualization",
                     width: int = 1400,
                     height: int = 900,
                     show_axes: bool = True,
                     show_grid: bool = True,
                     bgcolor: str = "rgb(245, 245, 250)") -> go.Figure:
        """Create interactive Plotly figure.
        
        :param title: Title for the figure
        :param width: Figure width in pixels
        :param height: Figure height in pixels
        :param show_axes: Show axis labels
        :param show_grid: Show grid lines
        :param bgcolor: Background color
        :return: Plotly figure object
        """
        fig = go.Figure()
        
        # Add point clouds
        for pc in self.point_clouds:
            points = pc.points
            
            # Subsample if needed
            if len(points) > self.max_points_display:
                idx = np.random.choice(len(points), self.max_points_display, replace=False)
                points = points[idx]
                colors = pc.colors[idx] if pc.colors is not None else None
            else:
                colors = pc.colors
            
            # Setup marker properties
            marker_dict = dict(
                size=pc.point_size,
                opacity=pc.opacity
            )
            
            # Special handling for correspondence points
            if "Correspondences" in pc.name:
                marker_dict['color'] = 'red'
            elif colors is not None:
                # Convert colors to plotly format
                marker_dict['color'] = ['rgb({},{},{})'.format(r, g, b) for r, g, b in colors]
            else:
                marker_dict['color'] = 'blue'
            
            # Add trace
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=marker_dict,
                name=pc.name,
                visible=pc.visible,
                hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}',
                text=[pc.name] * len(points)
            ))
        
        # Add cameras
        if self.cameras:
            camera_positions = np.array([cam.position for cam in self.cameras])
            camera_names = [cam.name for cam in self.cameras]
            camera_colors = [cam.color for cam in self.cameras]
            
            fig.add_trace(go.Scatter3d(
                x=camera_positions[:, 0],
                y=camera_positions[:, 1],
                z=camera_positions[:, 2],
                mode='markers',
                marker=dict(
                    size=self.cameras[0].size if self.cameras else 8,
                    color=camera_colors,
                    symbol='diamond'
                ),
                name='Cameras',
                hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}',
                text=camera_names
            ))
        
        # Add view control buttons
        self._add_view_controls(fig)
        
        # Compute axis ranges from COLMAP sparse points if available
        x_range = None
        y_range = None
        z_range = None
        
        for pc in self.point_clouds:
            if "COLMAP" in pc.name and "Sparse" in pc.name:
                points = pc.points
                if len(points) > 0:
                    margin = 0.1  # 10% margin
                    x_min, x_max = points[:, 0].min(), points[:, 0].max()
                    y_min, y_max = points[:, 1].min(), points[:, 1].max()
                    z_min, z_max = points[:, 2].min(), points[:, 2].max()
                    
                    x_margin = (x_max - x_min) * margin
                    y_margin = (y_max - y_min) * margin
                    z_margin = (z_max - z_min) * margin
                    
                    x_range = [x_min - x_margin, x_max + x_margin]
                    y_range = [y_min - y_margin, y_max + y_margin]
                    z_range = [z_min - z_margin, z_max + z_margin]
                break
        
        # Update layout with elegant styling
        scene_dict = dict(
            aspectmode='data',  # Use natural data proportions
            bgcolor=bgcolor,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        )
        
        # Configure axes
        axis_config = dict(
            backgroundcolor=bgcolor,
            gridcolor="rgba(200, 200, 200, 0.3)",
            showbackground=True,
            zerolinecolor="rgba(150, 150, 150, 0.5)",
            linecolor="rgba(100, 100, 100, 0.5)",
            linewidth=1,
            showspikes=False
        )
        
        # Add ranges if we have them from COLMAP
        if x_range is not None:
            axis_config['autorange'] = False
        
        if show_axes:
            xaxis_dict = dict(title='X', **axis_config)
            yaxis_dict = dict(title='Y', **axis_config)
            zaxis_dict = dict(title='Z', **axis_config)
            
            if x_range is not None:
                xaxis_dict['range'] = x_range
                yaxis_dict['range'] = y_range
                zaxis_dict['range'] = z_range
            
            scene_dict.update(dict(
                xaxis=xaxis_dict,
                yaxis=yaxis_dict,
                zaxis=zaxis_dict
            ))
        else:
            xaxis_dict = dict(showticklabels=False, title='', **axis_config)
            yaxis_dict = dict(showticklabels=False, title='', **axis_config)
            zaxis_dict = dict(showticklabels=False, title='', **axis_config)
            
            if x_range is not None:
                xaxis_dict['range'] = x_range
                yaxis_dict['range'] = y_range
                zaxis_dict['range'] = z_range
            
            scene_dict.update(dict(
                xaxis=xaxis_dict,
                yaxis=yaxis_dict,
                zaxis=zaxis_dict
            ))
        
        if not show_grid:
            for axis in ['xaxis', 'yaxis', 'zaxis']:
                scene_dict[axis].update(dict(
                    showgrid=False,
                    showbackground=False,
                    zeroline=False
                ))
        
        # Apply elegant layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family="Arial, sans-serif", color="rgb(50, 50, 50)"),
                x=0.5,
                xanchor='center'
            ),
            width=width,
            height=height,
            scene=scene_dict,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="right",
                x=0.98,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(100, 100, 100, 0.3)",
                borderwidth=1,
                font=dict(size=11, family="Arial")
            ),
            paper_bgcolor="rgb(255, 255, 255)",
            plot_bgcolor=bgcolor,
            margin=dict(l=0, r=0, t=50, b=0),
            hovermode='closest',
            dragmode='orbit'
        )
        
        return fig
    
    def _add_view_controls(self, fig: go.Figure) -> None:
        """Add interactive view control buttons to figure.
        
        :param fig: Plotly figure to add controls to
        """
        n_traces = len(fig.data)
        n_pointclouds = len(self.point_clouds)
        has_cameras = len(self.cameras) > 0
        
        # Categorize point clouds
        colmap_indices = []
        correspondence_indices = []
        moge_indices = []
        refined_indices = []
        other_indices = []
        
        for i, pc in enumerate(self.point_clouds):
            if "Correspondences" in pc.name:
                correspondence_indices.append(i)
            elif "COLMAP" in pc.name:
                colmap_indices.append(i)
            elif "MoGe" in pc.name or ("Initial" in pc.name and "RGBD" in pc.name):
                moge_indices.append(i)
            elif "Refined" in pc.name:
                refined_indices.append(i)
            else:
                other_indices.append(i)
        
        # === MAIN VIEWING OPTIONS (as requested by user) ===
        viewing_buttons = []
        
        # 1. Sparse Points + Image Corresp (Red)
        if colmap_indices and correspondence_indices:
            visible = [False] * n_pointclouds
            for idx in colmap_indices + correspondence_indices:
                visible[idx] = True
            if has_cameras:
                visible.append(False)
            viewing_buttons.append(dict(
                args=[{"visible": visible}],
                label="Sparse + Corresp (Red)",
                method="update"
            ))
        
        # 2. Initial MoGe
        if moge_indices:
            visible = [False] * n_pointclouds
            for idx in moge_indices:
                visible[idx] = True
            if has_cameras:
                visible.append(False)
            viewing_buttons.append(dict(
                args=[{"visible": visible}],
                label="Initial MoGe",
                method="update"
            ))
        
        # 3. Refined MoGe
        if refined_indices:
            visible = [False] * n_pointclouds
            for idx in refined_indices:
                visible[idx] = True
            if has_cameras:
                visible.append(False)
            viewing_buttons.append(dict(
                args=[{"visible": visible}],
                label="Refined MoGe",
                method="update"
            ))
        
        # 4. Corresp + Initial
        if correspondence_indices and moge_indices:
            visible = [False] * n_pointclouds
            for idx in correspondence_indices + moge_indices:
                visible[idx] = True
            if has_cameras:
                visible.append(False)
            viewing_buttons.append(dict(
                args=[{"visible": visible}],
                label="Corresp + Initial",
                method="update"
            ))
        
        # 5. Corresp + Refined
        if correspondence_indices and refined_indices:
            visible = [False] * n_pointclouds
            for idx in correspondence_indices + refined_indices:
                visible[idx] = True
            if has_cameras:
                visible.append(False)
            viewing_buttons.append(dict(
                args=[{"visible": visible}],
                label="Corresp + Refined",
                method="update"
            ))
        
        # === PRESET CONTROLS ===
        preset_buttons = []
        
        # === VIEW CONTROLS ===
        view_buttons = []
        
        view_buttons.append(dict(
            args=[{"scene.camera.projection.type": "orthographic"}],
            label="Orthographic",
            method="relayout"
        ))
        
        view_buttons.append(dict(
            args=[{"scene.camera.projection.type": "perspective"}],
            label="Perspective",
            method="relayout"
        ))
        
        view_buttons.append(dict(
            args=[{"scene.camera": dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )}],
            label="Reset View",
            method="relayout"
        ))
        
        # === CAMERA CONTROLS ===
        camera_buttons = []
        if has_cameras:
            # Find the current visible state for all point clouds
            visible_state = [pc.visible for pc in self.point_clouds]
            
            camera_buttons.append(dict(
                args=[{"visible": visible_state + [True]}],
                label="Show Cameras",
                method="update"
            ))
            camera_buttons.append(dict(
                args=[{"visible": visible_state + [False]}],
                label="Hide Cameras",
                method="update"
            ))
        
        # === BUILD UPDATE MENUS ===
        # Combine all buttons into a single vertical stack on the left
        updatemenus = []
        current_y = 0.99
        button_height = 0.035
        section_spacing = 0.02
        
        # Main viewing options (top)
        if viewing_buttons:
            updatemenus.append(dict(
                type="buttons",
                direction="down",
                buttons=viewing_buttons,
                pad={"r": 2, "t": 2, "b": 2, "l": 2},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=current_y,
                yanchor="top",
                bgcolor="rgba(245, 250, 255, 0.95)",
                bordercolor="rgba(100, 150, 200, 0.3)",
                borderwidth=1,
                font=dict(size=10, family="Arial")
            ))
            current_y -= len(viewing_buttons) * button_height + section_spacing
        
        # Preset controls (below viewing options)
        if preset_buttons:
            updatemenus.append(dict(
                type="buttons",
                direction="down",
                buttons=preset_buttons,
                pad={"r": 2, "t": 2, "b": 2, "l": 2},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=current_y,
                yanchor="top",
                bgcolor="rgba(240, 240, 245, 0.95)",
                bordercolor="rgba(60, 60, 60, 0.3)",
                borderwidth=1,
                font=dict(size=10, family="Arial", color="rgb(40, 40, 40)")
            ))
            current_y -= len(preset_buttons) * button_height + section_spacing
        
        # View controls (below preset controls)
        if view_buttons:
            updatemenus.append(dict(
                type="buttons",
                direction="down",
                buttons=view_buttons,
                pad={"r": 2, "t": 2, "b": 2, "l": 2},
                showactive=False,
                x=0.01,
                xanchor="left",
                y=current_y,
                yanchor="top",
                bgcolor="rgba(250, 250, 250, 0.95)",
                bordercolor="rgba(100, 100, 100, 0.3)",
                borderwidth=1,
                font=dict(size=10, family="Arial")
            ))
            current_y -= len(view_buttons) * button_height + section_spacing
        
        # Camera controls (bottom of stack)
        if camera_buttons:
            updatemenus.append(dict(
                type="buttons",
                direction="down",
                buttons=camera_buttons,
                pad={"r": 2, "t": 2, "b": 2, "l": 2},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=max(0.05, current_y),
                yanchor="top",
                bgcolor="rgba(250, 250, 250, 0.95)",
                bordercolor="rgba(100, 100, 100, 0.3)",
                borderwidth=1,
                font=dict(size=10, family="Arial")
            ))
        
        fig.update_layout(updatemenus=updatemenus)
    
    def save_html(self, filepath: Union[str, Path], 
                 auto_open: bool = False,
                 include_plotlyjs: Literal['cdn', 'directory', 'inline'] = 'cdn') -> None:
        """Save visualization as interactive HTML file.
        
        :param filepath: Path to save HTML file
        :param auto_open: Automatically open in browser
        :param include_plotlyjs: How to include plotly.js ('cdn', 'directory', or 'inline')
        """
        fig = self.create_figure()
        fig.write_html(
            str(filepath),
            auto_open=auto_open,
            include_plotlyjs=include_plotlyjs
        )
    
    def create_comparison_figure(self, 
                               pc_indices: list[int],
                               titles: Optional[list[str]] = None,
                               sync_cameras: bool = True) -> go.Figure:
        """Create side-by-side comparison of multiple point clouds.
        
        :param pc_indices: Indices of point clouds to compare
        :param titles: Optional titles for each subplot
        :param sync_cameras: Synchronize camera views across subplots
        :return: Plotly figure with subplots
        """
        n_plots = len(pc_indices)
        
        if titles is None:
            titles = [self.point_clouds[i].name for i in pc_indices]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=n_plots,
            specs=[[{'type': 'scatter3d'}] * n_plots],
            subplot_titles=titles
        )
        
        # Add point clouds to each subplot
        for col, idx in enumerate(pc_indices, start=1):
            pc = self.point_clouds[idx]
            points = pc.points
            
            # Subsample if needed
            if len(points) > self.max_points_display:
                sample_idx = np.random.choice(len(points), self.max_points_display, replace=False)
                points = points[sample_idx]
                colors = pc.colors[sample_idx] if pc.colors is not None else None
            else:
                colors = pc.colors
            
            # Setup marker properties
            marker_dict = dict(
                size=pc.point_size,
                opacity=pc.opacity
            )
            
            if colors is not None:
                marker_dict['color'] = ['rgb({},{},{})'.format(r, g, b) for r, g, b in colors]
            else:
                marker_dict['color'] = 'blue'
            
            # Add trace
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='markers',
                    marker=marker_dict,
                    name=pc.name,
                    showlegend=False
                ),
                row=1, col=col
            )
        
        # Update layout
        camera_dict = None
        if sync_cameras:
            # Use same camera for all subplots
            camera_dict = dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0)
            )
        
        for i in range(1, n_plots + 1):
            scene_name = f'scene{i}' if i > 1 else 'scene'
            scene_config = dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
            if camera_dict:
                scene_config['camera'] = camera_dict
            
            fig.update_layout(**{scene_name: scene_config})
        
        fig.update_layout(
            title="Point Cloud Comparison",
            width=400 * n_plots,
            height=600,
            showlegend=False
        )
        
        return fig


def create_quick_visualization(colmap_path: Union[str, Path],
                             depth_map: Optional[np.ndarray] = None,
                             rgb_image: Optional[np.ndarray] = None,
                             K: Optional[np.ndarray] = None,
                             cam_from_world: Optional[np.ndarray] = None,
                             output_path: Optional[Union[str, Path]] = None) -> go.Figure:
    """Quick helper function to create visualization.
    
    :param colmap_path: Path to COLMAP reconstruction
    :param depth_map: Optional depth map from MoGe
    :param rgb_image: Optional RGB image
    :param K: Camera intrinsics
    :param cam_from_world: Camera extrinsics
    :param output_path: Optional path to save HTML
    :return: Plotly figure
    """
    viz = COLMAPVisualizer()
    
    # Load COLMAP
    viz.load_colmap(colmap_path)
    
    # Add RGBD if provided
    if depth_map is not None and K is not None and cam_from_world is not None:
        viz.add_rgbd_pointcloud(
            depth_map=depth_map,
            rgb_image=rgb_image,
            K=K,
            cam_from_world=cam_from_world,
            name="MoGe Depth"
        )
    
    # Create figure
    fig = viz.create_figure()
    
    # Save if requested
    if output_path:
        viz.save_html(output_path)
    
    return fig