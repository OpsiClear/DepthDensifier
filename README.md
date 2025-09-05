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

## Getting Test Data

To download test datasets and set up the data directory structure, use one of the provided scripts:

### Python Script (Recommended)
```bash
# Download datasets and recommended MoGe model (default)
uv run python scripts/download_data.py

# Download only datasets, skip models
uv run python scripts/download_data.py --config.no-download-moge-models

# Download all MoGe models
uv run python scripts/download_data.py --config.moge-models all

# Download specific MoGe models
uv run python scripts/download_data.py --config.moge-models "moge-2-vitl,moge-2-vitb-normal"

# Download only MoGe v2 models
uv run python scripts/download_data.py --config.moge-models v2

# Custom data directory and options
uv run python scripts/download_data.py --config.data-dir my_data --config.keep-zip --config.skip-existing

# See all options
uv run python scripts/download_data.py --help
```

#### Available MoGe Models:
- **`recommended`** (default): `moge-2-vitl-normal` - Latest ViT-Large with metric scale and normal maps
- **`all`**: All available models (5 models, ~10GB total)
- **`v1`**: MoGe-1 models - `moge-vitl` (314M params)
- **`v2`**: MoGe-2 models - 4 variants with metric scale support
- **Custom**: Comma-separated model names for specific downloads

### Shell Scripts
```bash
# Linux/macOS
./scripts/download_data.sh

# Windows PowerShell
.\scripts\download_data.ps1

# Keep zip files after extraction
.\scripts\download_data.ps1 --keep-zip
```

This will create the following structure:
```
DepthDensifier/
├── data/            # Test data
│   ├── 360_v2/      # RefNeRF 360° test dataset
│   ├── outputs/     # Processing results
│   └── temp/        # Temporary files
└── models/          # Pretrained models
    └── moge/        # MoGe models folder
        ├── moge-2-vitl-normal/     # Recommended model (metric + normal)
        ├── moge-2-vitl/            # MoGe-2 ViT-Large
        ├── moge-2-vitb-normal/     # MoGe-2 ViT-Base
        ├── moge-2-vits-normal/     # MoGe-2 ViT-Small
        └── moge-vitl/              # MoGe-1 ViT-Large
```

### Using Downloaded MoGe Models

Once downloaded, you can use the models in your code:

```python
from moge.model.v2 import MoGeModel
import torch

# Load the recommended model
model = MoGeModel.from_pretrained("models/moge/moge-2-vitl-normal")

# Or load any other downloaded model
model = MoGeModel.from_pretrained("models/moge/moge-2-vitb-normal")
```

## Usage

*Coming soon - usage examples and API documentation*

## License

This project is licensed under the terms specified in the LICENSE file.
