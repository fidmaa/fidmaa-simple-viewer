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


def FIDMAA_to_pyvista_surface(image, depthmap):
    import pyvista

    depthmap = Image_to_OpenCV(depthmap.convert("L"))
    depthmap = np.flipud(depthmap)

    # Load an image to use as a texture
    colors = np.array(image)

    # Convert depth map to point cloud
    point_cloud = depth_map_to_point_cloud(depthmap)

    pdata = pyvista.PolyData(point_cloud, force_float=False)
    # Compute the surface mesh from the point cloud using the Delaunay triangulation
    surface = pdata.delaunay_2d()

    # Create a PyVista image object from the RGB image data
    image_pv = pyvista.pyvista_ndarray(colors)

    # Create a texture from the PyVista image object
    texture = pyvista.Texture(image_pv)

    # Map the texture onto the PolyData object
    surface.texture_map_to_plane(inplace=True, use_bounds=True)

    return surface, texture


def pyvista_show(surface, texture):
    # Plot the PolyData object with the texture
    import pyvista
    plotter = pyvista.Plotter(line_smoothing=True, )
    plotter.add_mesh(surface, texture=texture)
    plotter.add_text("FIDMAA (C) 2022-2024 Michal Pasternak & collaborators ")
    plotter.show()


def visualise_fidmaa_image(filename):
    image, depth_map = load_image(filename)
    surface, texture = FIDMAA_to_pyvista_surface(image, depth_map)
    pyvista_show(surface, texture)


@click.command()
@click.argument('input')
def display_fidmaa_image(input):
    visualise_fidmaa_image(input)


if __name__ == '__main__':
    display_fidmaa_image()
