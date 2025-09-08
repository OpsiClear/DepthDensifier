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
# Download datasets and all MoGe models (default)
uv run python downloads.py

# Download only datasets, skip models
uv run python downloads.py --config.no-download-moge-models

# Download only recommended MoGe model
uv run python downloads.py --config.moge-models recommended

# Download specific MoGe models
uv run python downloads.py --config.moge-models "moge-2-vitl,moge-2-vitb-normal"

# Download only MoGe v2 models
uv run python downloads.py --config.moge-models v2

# Custom data directory and options
uv run python downloads.py --config.data-dir my_data --config.keep-zip --config.skip-existing

# See all options
uv run python downloads.py --help
```

#### Available MoGe Models:
- **`all`** (default): All available models (5 models, ~10GB total)
- **`recommended`**: `moge-2-vitl-normal` - Latest ViT-Large with metric scale and normal maps
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

The download script intelligently detects existing data and skips downloads when appropriate. It will create the following structure:
```
DepthDensifier/
├── data/            # Test data
│   ├── 360_v2/      # RefNeRF 360° test dataset
│   ├── outputs/     # Processing results
│   └── temp/        # Temporary files
├── models/          # Pretrained models (downloaded by default)
│   └── moge/        # MoGe models folder
│       ├── moge-2-vitl-normal/     # Recommended model (metric + normal)
│       ├── moge-2-vitl/            # MoGe-2 ViT-Large
│       ├── moge-2-vitb-normal/     # MoGe-2 ViT-Base
│       ├── moge-2-vits-normal/     # MoGe-2 ViT-Small
│       └── moge-vitl/              # MoGe-1 ViT-Large
└── downloads.py     # Download script (moved to project root)
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

### Batch Processing All Datasets

The project includes scripts to automatically process all datasets in your `data/` directory using the depth densification pipeline.

#### PowerShell Script (Windows) - Recommended

```powershell
# Run on all datasets with default settings
.\scripts\run_all_datasets.ps1

# Or specify custom paths
.\scripts\run_all_datasets.ps1 -DataDir "data" -ResultsDir "results" -Script "scripts/run_pipeline.py"
```

#### Python Script (Cross-platform)

```bash
# On Windows
python scripts/run_batch_all.py

# On Linux/Mac
python3 scripts/run_batch_all.py
```

#### Individual Dataset Processing

To process a single dataset manually:

```bash
python scripts/run_pipeline.py --config.paths.recon-path "data/bicycle/sparse/0" --config.paths.image-dir "data/bicycle/images" --config.paths.output-model-dir "results/bicycle"
```

### What the Scripts Do

1. **Automatically discover datasets** in the `data/` directory
2. **Validate each dataset** has required COLMAP files (`cameras.bin`, `images.bin`, `points3D.bin`)
3. **Process each dataset sequentially** using the depth densification pipeline
4. **Save results** to separate directories under `results/`
5. **Provide progress tracking** and error reporting

### Dataset Structure Expected

Each dataset should have this structure:
```
data/
├── bicycle/
│   ├── images/          # Original images
│   ├── sparse/0/        # COLMAP reconstruction
│   │   ├── cameras.bin
│   │   ├── images.bin
│   │   └── points3D.bin
│   ├── images_2/        # Downsampled (optional)
│   ├── images_4/        # Downsampled (optional)
│   └── images_8/        # Downsampled (optional)
└── bonsai/
    └── ...
```

### Output Structure

Results will be saved as:
```
results/
├── bicycle/
│   └── 0/               # COLMAP model files
├── bonsai/
│   └── 0/
└── ...
```

### Processing Time

Each dataset may take **1-2 hours** to process depending on:
- Number of images
- Hardware (GPU vs CPU)
- Image resolution
- Processing parameters

### Error Handling

The scripts will:
- Skip datasets missing required files
- Continue processing other datasets if one fails
- Provide detailed error messages
- Generate a summary report at the end

### Configuration

To modify processing parameters, edit the `ProcessingConfig` and `FilteringConfig` classes in `scripts/run_pipeline.py`:

```python
@dataclass
class ProcessingConfig:
    pipeline_downsample_factor: int = 1      # 1 = original resolution, higher = faster
    downsample_density: int = 32             # Point cloud density (higher = more points)

@dataclass
class FilteringConfig:
    vote_threshold: int = 5                  # Multi-view consistency threshold
    depth_threshold: float = 0.7             # Depth filtering threshold
```

### Troubleshooting

1. **CUDA not available**: The script will warn but continue with CPU processing (slower)
2. **Missing dependencies**: Ensure all required packages are installed with `uv sync`
3. **Timeout**: Large datasets may need more time - the timeout is set to 2 hours per dataset
4. **Memory issues**: Reduce `pipeline_downsample_factor` or `downsample_density` in the config
5. **Git LFS issues**: The installation automatically handles LFS download errors by skipping missing files

## License

This project is licensed under the terms specified in the LICENSE file.
