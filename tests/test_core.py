from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from fidmaa_simple_viewer.core import (
    FIDMAA_to_pyvista_surface,
    depth_map_to_point_cloud,
    display_fidmaa_image,
)
from portrait_analyser.ios import load_image


@pytest.fixture
def heic_image_path():
    return Path(__file__).parent / "heic_depth_data.heic"


def test_depth_map_to_point_cloud():
    depth = np.array([[1, 2], [3, 4]])
    points = depth_map_to_point_cloud(depth)
    assert points.shape == (4, 3)
    assert np.array_equal(points[0], [0, 0, 1])
    assert np.array_equal(points[3], [1, 1, 4])


def test_fidmaa_to_pyvista_surface(heic_image_path):
    portrait = load_image(str(heic_image_path))
    surface, texture = FIDMAA_to_pyvista_surface(
        portrait.photo, portrait.depthmap,
        portrait.floatValueMin, portrait.floatValueMax,
    )
    assert surface.n_points > 0
    assert surface.n_faces_strict > 0
    assert texture is not None


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(display_fidmaa_image, ['--help'])
    assert result.exit_code == 0
    assert '--depth-scale' in result.output
    assert '--depth-cutoff' in result.output
    assert '--no-texture' in result.output
