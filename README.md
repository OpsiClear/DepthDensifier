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

### Available Pipelines

DepthDensifier offers two point sampling strategies:

1. **Grid-based sampling** - Traditional uniform grid sampling across images
2. **Intelligent point sampling** - Edge-aware sampling that focuses on high-detail regions

### Single Dataset Processing

#### Grid-Based Sampling (Original)

```bash
# Basic usage with default settings
uv run python scripts/run_pipeline.py

# Custom dataset paths
uv run python scripts/run_pipeline.py \
  --paths.recon-path "data/360_v2/bicycle/sparse/0" \
  --paths.image-dir "data/360_v2/bicycle/images" \
  --paths.output-model-dir "results/bicycle"

# Performance tuning with dense sampling
uv run python scripts/run_pipeline.py \
  --processing.batch-size 8 \
  --processing.downsample-density 8 \
  --processing.pipeline-downsample-factor 1
```

#### Intelligent Point Sampling

```bash
# Basic usage with edge-aware sampling
uv run python scripts/run_pipeline_point_sampler.py

# Custom sampling parameters
uv run python scripts/run_pipeline_point_sampler.py \
  --sampling.num-points-per-image 10000 \
  --sampling.sampling-strategy mixed \
  --sampling.edge-weight 0.8

# Different sampling strategies
# Pure random sampling
uv run python scripts/run_pipeline_point_sampler.py \
  --sampling.sampling-strategy random

# Pure edge-based sampling
uv run python scripts/run_pipeline_point_sampler.py \
  --sampling.sampling-strategy edges \
  --sampling.num-points-per-image 5000

# Mixed strategy (default)
uv run python scripts/run_pipeline_point_sampler.py \
  --sampling.sampling-strategy mixed \
  --sampling.edge-weight 0.7
```

### Batch Processing All Datasets

Process all valid datasets in your data directory:

#### Grid-Based Batch Processing

```bash
# Process all datasets with default settings
uv run python scripts/run_batch_all.py

# Custom directories and parameters
uv run python scripts/run_batch_all.py \
  --data-dir "data/360_v2" \
  --results-dir "results_grid" \
  --processing.downsample-density 16 \
  --processing.batch-size 4
```

#### Point Sampler Batch Processing

```bash
# Process all datasets with point sampling
uv run python scripts/run_batch_all_point_sampler.py

# Custom sampling configuration
uv run python scripts/run_batch_all_point_sampler.py \
  --data-dir "data/360_v2" \
  --results-dir "results_sampled" \
  --sampling.num-points-per-image 15000 \
  --sampling.sampling-strategy edges \
  --sampling.edge-weight 0.9
```

### Configuration Parameters

**Processing Parameters:**
- `pipeline_downsample_factor`: Image downsampling before processing (default: 1, no downsampling)
- `downsample_density`: Point cloud density control for grid sampling (default: 16, lower=denser)
- `batch_size`: GPU batch size for MoGe inference (default: 4)
- `skip_every_n_images`: Process every Nth image (default: 1, process all)
- `gpu_cache_clear_interval`: Clear GPU cache every N batches (default: 10)

**Point Sampling Parameters (point_sampler only):**
- `num_points_per_image`: Number of points to sample per image (default: 5000)
- `sampling_strategy`: Strategy for point selection (default: "mixed")
  - `"random"`: Uniform random sampling
  - `"edges"`: Sample based on image gradients
  - `"mixed"`: Combination of random and edge-based
- `edge_weight`: Weight for edge-based sampling in mixed mode (default: 0.7, range: 0-1)
- `use_gpu`: Use GPU acceleration for sampling (default: true)
- `seed`: Random seed for reproducible sampling (default: None)

**Refinement Parameters:**
- `use_fp16`: Enable FP16 precision for 2x speed (default: false)
- `skip_smoothing`: Skip median filtering for speed (default: false)
- `adaptive_correspondences`: Reduce correspondences for simple scenes (default: false)
- `min_correspondences`: Minimum sparse points required (default: 10)

**Filtering Parameters:**
- `vote_threshold`: Minimum votes to classify as floater (default: 5)
- `depth_threshold`: Depth consistency threshold (default: 0.9)
- `grazing_angle_threshold`: Maximum grazing angle in degrees (default: 85.0)
- `camera_distance_threshold`: Maximum camera distance for voting (default: 10.0)
- `max_cameras_to_check`: Maximum cameras to check per point (default: 20)

## Dataset Structure

Expected COLMAP reconstruction format:
```
data/
├── 360_v2/              # Test datasets directory
│   ├── bicycle/
│   │   ├── images/      # Original images
│   │   └── sparse/0/    # COLMAP reconstruction
│   │       ├── cameras.bin
│   │       ├── images.bin
│   │       └── points3D.bin
│   ├── bonsai/
│   │   └── ...
│   └── garden/
│       └── ...
└── outputs/             # Temporary processing files
```

Output structure:
```
results/                 # Grid-based sampling outputs
├── bicycle/
│   └── sparse/0/        # Dense COLMAP model
└── bonsai/
    └── sparse/0/

results_point_sampler/   # Point sampling outputs  
├── bicycle/
│   └── sparse/0/
└── bonsai/
    └── sparse/0/
```

## Performance Tips

1. **GPU Memory Management:**
   - Reduce `batch_size` if encountering OOM errors
   - Increase `gpu_cache_clear_interval` for better memory management
   - Use `pipeline_downsample_factor` > 1 for faster processing

2. **Speed vs Quality Trade-offs:**
   - Grid sampling: Lower `downsample_density` for denser but slower results
   - Point sampling: Increase `num_points_per_image` for higher quality
   - Use `skip_every_n_images` > 1 for faster but less complete reconstruction

3. **Sampling Strategy Selection:**
   - Use `"edges"` for scenes with clear structures and boundaries
   - Use `"random"` for uniformly textured scenes
   - Use `"mixed"` (default) for balanced results

## MoGe Model Selection

Download specific MoGe models based on your needs:

```bash
# Download both models (default)
uv run python downloads.py --config.moge-models all

# Download model with normal maps (recommended)
uv run python downloads.py --config.moge-models recommended

# Download depth-only model (faster, no normals)
uv run python downloads.py --config.moge-models depth-only

# Skip model download (datasets only)
uv run python downloads.py --config.no-download-moge-models
```
