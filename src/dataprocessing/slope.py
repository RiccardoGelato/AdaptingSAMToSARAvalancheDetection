import math
import numpy as np
import rasterio

def calculate_slope(dem_data, transform = None):
    """
    Calculate the slope in degrees for a given DEM.
    
    Parameters:
      dem_data: 2D numpy array with elevation values.
      transform: Affine transform of the raster to extract pixel size.
      
    Returns:
      slope_degrees: 2D numpy array of slope in degrees.
    """
    if transform is None:
        # If no transform is provided, assume a default pixel size of 10x10
        transform = rasterio.Affine(10, 0, 0, 0, -10, 0)
    # Calculate the pixel resolution in x and y from the affine transform.
    # For most north-up images, transform.a is pixel width and transform.e is pixel height.
    xres = math.sqrt(transform.a**2 + transform.b**2)
    yres = math.sqrt(transform.d**2 + transform.e**2)
    
    # Compute gradients. np.gradient returns [gradient_y, gradient_x]
    grad_y, grad_x = np.gradient(dem_data, yres, xres)
    
    # Calculate slope in radians: slope = arctan(sqrt(grad_x^2 + grad_y^2))
    slope_radians = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    
    # Convert slope to degrees.
    slope_degrees = np.degrees(slope_radians)
    return slope_degrees