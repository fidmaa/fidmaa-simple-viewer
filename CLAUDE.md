# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

fidmaa-simple-viewer is a Python CLI tool that visualizes HEIC images with embedded depth data as interactive 3D textured surfaces. It loads iOS portrait-mode photos (HEIC with depth maps), converts depth data to point clouds, triangulates a surface mesh, and renders it with PyVista.

## Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_core.py::test_fidmaa_to_pyvista_surface

# Run the CLI tool
uv run display_fidmaa_image <heic_file>
```

## System Dependencies

Requires `libheif-dev` and `libde265-dev` for HEIC support (on Ubuntu/Debian: `apt install libheif-dev libde265-dev`).

## Architecture

Single-module project in `src/fidmaa_simple_viewer/core.py`. The processing pipeline:

1. `portrait_analyser.ios.load_image()` extracts the RGB image and depth map from a HEIC file
2. `FIDMAA_to_pyvista_surface()` converts the depth map to a 3D point cloud, runs Delaunay 2D triangulation via PyVista, and maps the image as a texture
3. `pyvista_show()` renders the textured surface in an interactive 3D viewer

CLI entry point is `display_fidmaa_image` (Click command), registered in `pyproject.toml` under `[project.scripts]`.

## Testing

Tests use pytest with a sample HEIC fixture file at `tests/heic_depth_data.heic`. The `pytest.ini` sets `python_paths = src` for import resolution.
