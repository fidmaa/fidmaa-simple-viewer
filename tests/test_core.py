from pathlib import Path

import pytest

from fidmaa_simple_viewer.core import FIDMAA_to_pyvista_surface
from portrait_analyser.ios import load_image


@pytest.fixture
def heic_image_path():
    return Path(__file__).parent / "heic_depth_data.heic"

def test_fidmaa_to_pyvista_surface(heic_image_path):
    image, depthmap = load_image(str(heic_image_path))
    ret = FIDMAA_to_pyvista_surface(image, depthmap)
    assert ret