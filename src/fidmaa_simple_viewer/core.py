import click

import numpy as np
from PIL.Image import Image

from portrait_analyser.ios import load_image


def depth_map_to_point_cloud(depth_map):
    rows, cols = depth_map.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    return np.column_stack([xx.ravel(), yy.ravel(), depth_map.ravel()])


def Image_to_OpenCV(image: Image):
    return np.array(image)


def FIDMAA_to_pyvista_surface(image, depthmap, float_min, float_max, depth_scale: float = 13.0, depth_cutoff: float | None = None):
    import pyvista

    depthmap = Image_to_OpenCV(depthmap.convert("L"))
    depthmap = np.flipud(depthmap).astype(np.float64)

    # Rescale depth values from 0-255 grayscale to real camera distances
    depthmap = 100.0 / (float_max * depthmap / 255.0 + float_min * (1.0 - depthmap / 255.0))
    depthmap = depthmap * depth_scale

    if depth_cutoff is not None:
        threshold = np.percentile(depthmap, depth_cutoff * 100)
        depthmap[depthmap > threshold] = np.nan

    # Load an image to use as a texture
    colors = np.array(image)

    # Create a structured grid from the regular depth map pixel grid
    rows, cols = depthmap.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    grid = pyvista.StructuredGrid(
        xx.astype(np.float64), yy.astype(np.float64), depthmap,
    )
    surface = grid.extract_surface(algorithm=None)

    # Create a PyVista image object from the RGB image data
    image_pv = pyvista.pyvista_ndarray(colors)

    # Create a texture from the PyVista image object
    texture = pyvista.Texture(image_pv)

    # Map the texture onto the PolyData object
    surface.texture_map_to_plane(inplace=True, use_bounds=True)

    return surface, texture


def pyvista_show(image, depthmap, float_min, float_max, depth_scale=13.0, depth_cutoff=0.6, no_texture=False, title="FIDMAA"):
    import pyvista

    def _build_surface(scale, cutoff):
        cutoff_val = cutoff if cutoff < 1.0 else None
        s, t = FIDMAA_to_pyvista_surface(
            image, depthmap, float_min, float_max,
            depth_scale=scale, depth_cutoff=cutoff_val,
        )
        # Center mesh at origin so rotation pivot is at the midpoint
        s.translate(-np.array(s.center), inplace=True)
        return s, t

    surface, texture = _build_surface(depth_scale, depth_cutoff)

    plotter = pyvista.Plotter(line_smoothing=True, title=title)

    params = {
        'depth_scale': depth_scale,
        'depth_cutoff': depth_cutoff,
        'no_texture': no_texture,
    }

    def _add_mesh(surface, texture):
        if params['no_texture']:
            plotter.add_mesh(surface, color='gray', name='depth_surface')
        else:
            plotter.add_mesh(surface, texture=texture, name='depth_surface')

    camera_light = pyvista.Light(light_type='camera light')

    _add_mesh(surface, texture)
    if no_texture:
        plotter.add_light(camera_light)

    plotter.add_text("FIDMAA (C) 2024-2026 Michal Pasternak & collaborators ")
    plotter.add_axes(viewport=(0.7, 0.0, 1.0, 0.3))

    def _rebuild():
        s, t = _build_surface(params['depth_scale'], params['depth_cutoff'])
        _add_mesh(s, t)
        plotter.reset_camera()

    def on_depth_scale(value):
        params['depth_scale'] = value
        _rebuild()

    def on_depth_cutoff(value):
        params['depth_cutoff'] = value
        _rebuild()

    def on_texture_toggle(value):
        params['no_texture'] = not value
        plotter.renderer.RemoveAllLights()
        if not value:
            plotter.add_light(camera_light)
        _rebuild()

    plotter.add_slider_widget(
        on_depth_scale,
        rng=[0.0, 30.0],
        value=params['depth_scale'],
        title="Depth Scale",
        pointa=(0.025, 0.1),
        pointb=(0.31, 0.1),
        style='modern',
        interaction_event='always',
    )

    plotter.add_slider_widget(
        on_depth_cutoff,
        rng=[0.0, 1.0],
        value=params['depth_cutoff'],
        title="Depth Cutoff",
        pointa=(0.025, 0.25),
        pointb=(0.31, 0.25),
        style='modern',
        interaction_event='always',
    )

    plotter.add_checkbox_button_widget(
        on_texture_toggle,
        value=not params['no_texture'],
        position=(25, 280),
        size=25,
        border_size=3,
        color_on='green',
        color_off='grey',
    )
    plotter.add_text("Texture", position=(60, 283), font_size=10, name='texture_label')

    plotter.show()


def visualise_fidmaa_image(filename, depth_scale: float = 13.0, depth_cutoff: float = 0.6, no_texture: bool = False):
    import os
    portrait = load_image(filename)
    pyvista_show(
        portrait.photo, portrait.depthmap,
        portrait.floatValueMin, portrait.floatValueMax,
        depth_scale=depth_scale, depth_cutoff=depth_cutoff,
        no_texture=no_texture,
        title=os.path.basename(filename),
    )


@click.command()
@click.argument('input')
@click.option('--depth-scale', default=13.0, type=float, help='Depth scale multiplier for Z axis exaggeration.')
@click.option('--depth-cutoff', default=0.6, type=float, help='Percentile cutoff (0.0-1.0) to remove background. E.g. 0.9 keeps the closest 90% of points.')
@click.option('--no-texture', is_flag=True, default=False, help='Disable texture; show surface with uniform color and camera lighting.')
def display_fidmaa_image(input, depth_scale, depth_cutoff, no_texture):
    visualise_fidmaa_image(input, depth_scale=depth_scale, depth_cutoff=depth_cutoff, no_texture=no_texture)


if __name__ == '__main__':
    display_fidmaa_image()
