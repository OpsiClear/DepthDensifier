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

### 3. **Dense Point Cloud Generation**
   - Unprojects refined depth maps to 3D space
   - Generates dense points with colors and normals
   - Configurable density control via downsampling parameters

### 4. **Multi-View Consistency Filtering**
   - Filters based on depth consistency and grazing angle thresholds
   - Projects points across multiple views to detect floaters
   - Uses voting mechanism to identify and remove floaters

## Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU
- Git with Git LFS support

### Install with uv

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install uv following the instructions at: https://docs.astral.sh/uv/getting-started/installation/

Once uv is installed:

```bash
# Clone the repository
git clone <your-repo-url>
cd DepthDensifier
uv sync
```

Download test datasets and MoGe models:

```bash
# Download datasets and all MoGe models (default)
uv run downloads.py
```

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
