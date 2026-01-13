import numpy as np
#import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import rasterio
import sys
import tifffile as tiff

def read_rcs_image(image_path):
    """
    Read a tiff image and return it as a numpy array
    """
    img = tiff.imread(image_path)
    # Convert the image to a numpy array (if not already)
    image_array = np.array(img, dtype=np.float32)
    return image_array

def _rescale(arr, lo, hi):

    arr = (arr - lo) / (hi - lo)

    arr[arr < 0] = 0

    arr[arr > 1] = 1

    arr[np.isnan(arr)] = 0

    return arr

def _merge_to_rgb_all(hv0, hv1, vv0, vv1):

    hv0 = _rescale(hv0, -27, -7)

    hv1 = _rescale(hv1, -27, -7)

    vv0 = _rescale(vv0, -23, -3)

    vv1 = _rescale(vv1, -23, -3)

    a = _rescale(hv1 - hv0, 0, .25)

    b = _rescale(vv1 - vv0, 0, .25)

    w = _rescale(a - b, 0, 1)


    r = w*hv0 + (1 - w)*vv0

    g = w*hv1 + (1 - w)*vv1

    b = w*hv0 + (1 - w)*vv0
    

    rgb = np.stack([r, g, b], axis=2)

    result = []
    result.append(rgb)

    return result

def _merge_to_rgb_separate(hv0, hv1, vv0, vv1):

    hv0 = _rescale(hv0, -27, -7)

    hv1 = _rescale(hv1, -27, -7)

    vv0 = _rescale(vv0, -23, -3)

    vv1 = _rescale(vv1, -23, -3)    

    rgb_vv = np.stack([vv0, vv1, vv0], axis=2)
    rgb_hv = np.stack([hv0, hv1, hv0], axis=2)

    result = []
    result.append(rgb_vv)
    result.append(rgb_hv)

    return result

def _merge_to_rgb_difference(hv0, hv1, vv0, vv1):

    hv0 = _rescale(hv0, -27, -7)

    hv1 = _rescale(hv1, -27, -7)

    vv0 = _rescale(vv0, -23, -3)

    vv1 = _rescale(vv1, -23, -3)

    a = _rescale(hv1 - hv0, 0, .25)

    b = _rescale(vv1 - vv0, 0, .25)
    

    rgb_diff = np.stack([b,a,b], axis=2)

    result = []
    result.append(rgb_diff)

    return result

def _merge_all(hv0, hv1, vv0, vv1):

    hv0 = _rescale(hv0, -27, -7)

    hv1 = _rescale(hv1, -27, -7)

    vv0 = _rescale(vv0, -23, -3)

    vv1 = _rescale(vv1, -23, -3)
    

    output = np.stack([hv0, hv1, vv0, vv1, vv1], axis=2)

    result = []
    result.append(output)

    return result