import numpy as np
import SimpleITK as sitk
import pyvista as pv



itk_image = sitk.ReadImage('./data/test_dataset/1/unsegmented_Volume.mhd')

# vis using pyvista
vol = sitk.GetArrayFromImage(itk_image)


# Create a PyVista volume
volume = pv.wrap(vol)

# Visualize the volume
plotter = pv.Plotter()

# Add the volume to the plotter
plotter.add_volume(volume, cmap='gray')

# Add a sphere at the origin
plotter.add_mesh(pv.Sphere(radius=3, center=(1, 0, 0)))

#Add custom axes
axes_length = 50
plotter.add_mesh(pv.Line([0, 0, 0], [axes_length, 0, 0]), color='r')  # x-axis
plotter.add_mesh(pv.Line([0, 0, 0], [0, axes_length, 0]), color='g')  # y-axis
plotter.add_mesh(pv.Line([0, 0, 0], [0, 0, axes_length]), color='b')  # z-axis


# Visualize the volume
plotter.show()

