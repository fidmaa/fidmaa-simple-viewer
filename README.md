# fidmaa-simple-viewer

CLI tool that visualizes iOS portrait-mode HEIC photos as interactive 3D textured surfaces using PyVista.

## Installation

### System dependencies

On Debian/Ubuntu:

```bash
apt install libheif-dev libde265-dev
```

### Install from PyPI

```bash
pip install fidmaa-simple-viewer
```

## Usage

```bash
display_fidmaa_image photo.heic
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--depth-scale` | float | 13.0 | Depth scale multiplier for Z axis exaggeration |
| `--depth-cutoff` | float | 0.6 | Percentile cutoff (0.0-1.0) to remove background. E.g. 0.9 keeps the closest 90% of points |
| `--no-texture` | flag | off | Disable texture; show surface with uniform color and camera lighting |

### Interactive viewer controls

The 3D viewer provides runtime sliders for depth scale and depth cutoff, plus a checkbox to toggle texture on/off.

## Development

```bash
git clone https://github.com/fidmaa/fidmaa-simple-viewer.git
cd fidmaa-simple-viewer
uv sync
```

Run tests:

```bash
uv run pytest
```

### Architecture

Single-module project in `src/fidmaa_simple_viewer/core.py`. The pipeline:

1. `portrait_analyser.ios.load_image()` — extracts the RGB image and depth map from a HEIC file
2. `FIDMAA_to_pyvista_surface()` — converts the depth map to a 3D point cloud and runs Delaunay 2D triangulation
3. `pyvista_show()` — renders the textured surface in an interactive 3D viewer

## Requirements

Python >=3.12, <3.13

## License

MIT
