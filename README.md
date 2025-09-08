# DepthDensifier

Densify COLMAP sparse point clouds using monocular depth estimation with Microsoft's MoGe (Monocular Geometry).

## Pipeline Strategy

The DepthDensifier pipeline employs a multi-stage approach to transform sparse COLMAP reconstructions into dense point clouds:

### 1. **Depth Estimation with MoGe**
   - Uses Microsoft's MoGe v2 models to predict metric-scale depth maps from single images
   - Generates high-quality depth and normal maps for each input image
   - Supports multiple model variants (ViT-Small/Base/Large) with different speed/quality tradeoffs

### 2. **Depth Refinement with PCHIP Interpolation**
   - Aligns MoGe depth predictions with COLMAP's sparse 3D points
   - Uses GPU-accelerated PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) for monotonic depth correction
   - Preserves depth ordering while maintaining local consistency
   - Optional FP16 processing for 2x speedup on modern GPUs

### 3. **Dense Point Cloud Generation**
   - Unprojects refined depth maps to 3D space
   - Generates dense points with colors and normals
   - Configurable density control via downsampling parameters

### 4. **Multi-View Consistency Filtering**
   - Projects points across multiple views to detect "floaters" (inconsistent points)
   - Uses voting mechanism to identify and remove geometric inconsistencies
   - Filters based on depth consistency and grazing angle thresholds

### 5. **Optimizations**
   - **Batch Processing**: Processes multiple images simultaneously on GPU
   - **Async I/O**: Prefetches images while GPU processes current batch
   - **Memory Management**: Pre-allocates arrays and clears GPU cache periodically
   - **Vectorized Operations**: Uses NumPy/PyTorch vectorization throughout

## Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended for PyTorch acceleration)
- Git with Git LFS support

### Install with uv

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install uv following the instructions at: https://docs.astral.sh/uv/getting-started/installation/

Once uv is installed:

```bash
# Clone the repository
git clone <your-repo-url>
cd DepthDensifier

# Install the project and all dependencies
uv sync
```

## Dependencies

**Core Libraries:**
- PyTorch with CUDA support (CUDA 12.8)
- PyColmap for COLMAP integration
- NumPy, SciPy for numerical computing
- Pillow for image processing

**3D Processing:**
- MoGe (Microsoft Monocular Geometry) - installed from GitHub
- Matplotlib, Plotly for visualization

**Pipeline Tools:**
- Tyro for CLI configuration
- Hugging Face Hub for model downloads
- Numba for JIT compilation
- Scikit-learn for utilities

## Getting Test Data

Download test datasets and MoGe models:

```bash
# Download datasets and all MoGe models (default)
uv run python downloads.py

# Download only recommended model (moge-2-vitl-normal)
uv run python downloads.py --config.moge-models recommended

# Custom options
uv run python downloads.py --config.data-dir my_data --config.skip-existing

# See all options
uv run python downloads.py --help
```

### Available MoGe Models:
- **`recommended`**: `moge-2-vitl-normal` - ViT-Large with metric scale and normal maps
- **`v1`**: MoGe-1 models (relative depth only)
- **`v2`**: MoGe-2 models with metric scale support (4 variants)
- **`all`**: All available models (~10GB total)

## Usage

### Running the Pipeline

Process a single dataset:

```bash
# Basic usage with default settings
uv run scripts/run_pipeline_optimized_v2.py

# Custom dataset paths
uv run scripts/run_pipeline_optimized_v2.py \
  --paths.recon-path "data/bicycle/sparse/0" \
  --paths.image-dir "data/bicycle/images" \
  --paths.output-model-dir "results/bicycle"

# Performance tuning
uv run scripts/run_pipeline_optimized_v2.py \
  --processing.batch-size 8 \
  --processing.downsample-density 16 \
  --refiner.use-fp16 true \
  --refiner.skip-smoothing true
```

### Batch Processing All Datasets

Process all datasets in your `data/` directory:

```bash
# Python script (cross-platform)
uv run scripts/run_batch_all.py

# PowerShell script (Windows)
.\scripts\run_all_datasets.ps1
```

### Configuration Parameters

**Processing Parameters:**
- `pipeline_downsample_factor`: Image downsampling before processing (1=original)
- `downsample_density`: Point cloud density control (lower=denser)
- `batch_size`: GPU batch size for inference
- `prefetch_batches`: Number of batches to prefetch

**Refinement Parameters:**
- `use_fp16`: Enable FP16 for 2x speed
- `skip_smoothing`: Skip median filtering for maximum speed
- `adaptive_correspondences`: Reduce correspondences for simple scenes
- `min_correspondences`: Minimum sparse points required

**Filtering Parameters:**
- `vote_threshold`: Votes needed to remove floaters (default: 5)
- `depth_threshold`: Depth consistency threshold (default: 0.9)
- `grazing_angle_threshold`: Angle threshold for filtering

## Dataset Structure

Expected COLMAP reconstruction format:
```
data/
├── bicycle/
│   ├── images/          # Original images
│   └── sparse/0/        # COLMAP reconstruction
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── bonsai/
    └── ...
```

Output structure:
```
results/
├── bicycle/
│   └── 0/               # Dense COLMAP model
└── bonsai/
    └── 0/
```

## Performance Tips

1. **GPU Memory**: Reduce `batch_size` if running out of VRAM
2. **Speed**: Enable `use_fp16` and `skip_smoothing` for faster processing
3. **Quality**: Lower `downsample_density` for denser points (but slower)
4. **Large Datasets**: Increase `gpu_cache_clear_interval` to reduce memory fragmentation

## Troubleshooting

- **CUDA not available**: Will fallback to CPU (much slower)
- **Memory issues**: Reduce batch size or increase downsample density
- **Git LFS errors**: Installation auto-skips LFS files, core functionality remains intact
- **Missing dependencies**: Run `uv sync` to ensure all packages are installed

## License

This project is licensed under the terms specified in the LICENSE file.