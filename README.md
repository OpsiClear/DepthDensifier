# DepthDensifier

Densify a COLMAP point cloud using depth maps with Microsoft's MoGe (Monocular Geometry) integration.

## Features

- COLMAP point cloud densification using depth estimation
- Integration with Microsoft MoGe for monocular geometry estimation
- FastAPI web interface for easy interaction
- Modern Python packaging with `src/` layout

## Installation

### Prerequisites

- Python 3.12+
- Git with Git LFS support
- CUDA-compatible GPU (recommended for PyTorch acceleration)

### Quick Install

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd DepthDensifier
   ```

2. **Install with uv (recommended):**
   ```bash
   # Install uv if you haven't already
   pip install uv
   
   # Install the project and all dependencies
   uv sync
   ```

3. **Alternative: Install with pip:**
   ```bash
   pip install -e .
   ```

### Dependencies

This project includes several key dependencies:

- **Core ML/CV Libraries:**
  - PyTorch with CUDA support
  - OpenCV for computer vision
  - Pillow for image processing
  - NumPy, SciPy for numerical computing

- **3D Processing:**
  - MoGe (Microsoft Monocular Geometry) - installed from GitHub
  - utils3d for 3D utilities
  - trimesh for mesh processing

- **Web Interface:**
  - FastAPI for REST API
  - Gradio for interactive web UI
  - Uvicorn as ASGI server

### Git LFS Note

The MoGe dependency uses Git LFS for large files. If you encounter LFS download issues, the installation automatically skips LFS files during dependency resolution. The core functionality will work without these files, which are typically example data and pre-trained models.

### Development Setup

For development, the project uses a modern `src/` layout:

```
DepthDensifier/
├── src/
│   └── depthdensifier/     # Main package
├── pyproject.toml          # Project configuration
├── README.md
└── LICENSE
```

### Verification

After installation, verify everything works:

```python
import depthdensifier
print(f"DepthDensifier version: {depthdensifier.__version__}")

# Test MoGe integration
import moge
print("MoGe successfully imported!")
```

## Usage

*Coming soon - usage examples and API documentation*

## License

This project is licensed under the terms specified in the LICENSE file.
