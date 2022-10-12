
# Import the necessary packages
import numpy as np
from matplotlib import pyplot as plt

from skimage import io
from skimage import img_as_ubyte, img_as_uint, img_as_float64

from skimage import data,exposure
from skimage.transform import rotate

# from sklearn.cluster import KMeans
import argparse

def plot_channel_intensities(image):

    # plot the distribution of intensities for each of the three bands in the
    # image.

    red = image[:,:,0]
    green = image[:,:,1]
    blue = image[:,:,2]

    mask = ( red + green + blue ) > 0

    fig, ax_hist = plt.subplots(1, 1, figsize=(6, 5))

    # histogram of each
    ax_hist.hist(red[mask].ravel(), bins=256, histtype='step', color='red')
    ax_hist.hist(green[mask].ravel(), bins=256, histtype='step', color='green')
    ax_hist.hist(blue[mask].ravel(), bins=256, histtype='step', color='blue')

    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    ax_hist.set_ylabel('Pixel Count')
    ax_hist.set_xlabel('Pixel intensity')

def rescale_intensities(image,flow,fhigh):

    # Scale the intensities for each channel of an R B G image given an upper
    # and lower percentaile and return the re-scaled image.

    red = image[:,:,0]
    green = image[:,:,1]
    blue = image[:,:,2]

    mask = ( red + green + blue ) > 0

    # Pick percentile overwhich to rescale intensities
    #flow = 2
    #fhigh = 98

    plow, phigh = np.percentile(red[mask], (flow, fhigh))
    ls_red_rs= exposure.rescale_intensity(red, in_range=(plow, phigh))

    plow, phigh = np.percentile(green[mask], (flow, fhigh))
    ls_green_rs= exposure.rescale_intensity(green, in_range=(plow, phigh))

    plow, phigh = np.percentile(blue[mask], (flow, fhigh))
    ls_blue_rs= exposure.rescale_intensity(blue, in_range=(plow, phigh))

    # Define empty image
    nx,ny = ls_red_rs.shape
    ls_rgb_stretched = np.zeros([nx,ny,3],dtype=np.float64)

    # set all 3 channels of the image
    ls_rgb_stretched[:,:,0] = ls_red_rs
    ls_rgb_stretched[:,:,1] = ls_green_rs
    ls_rgb_stretched[:,:,2] = ls_blue_rs

    return  ls_rgb_stretched

def prepare_landsat_image(bands):

    # prepare a false color image for a combination of 3 landsat bands
    # Note this is specific to the set of images we provided, and the
    # hardcoded values for rotating need to be changed if you download a new
    # set of landsat images.

    ls_red = io.imread('landsat_band'+str(bands[0])+'.tif') # red
    ls_green = io.imread('landsat_band'+str(bands[1])+'.tif') # green
    ls_blue = io.imread('landsat_band'+str(bands[2])+'.tif') # blue

    ls_red = img_as_float64(ls_red)
    ls_green = img_as_float64(ls_green)
    ls_blue = img_as_float64(ls_blue)

    nx,ny = ls_red.shape
    ls_false = np.zeros([nx,ny,3],dtype=np.float64)

    ls_false[:,:,0] = ls_red
    ls_false[:,:,1] = ls_green
    ls_false[:,:,2] = ls_blue

    flow = 2
    fhigh = 98

    ls_false_rescale = rescale_intensities(ls_false,flow,fhigh)

    angle = 12.8
    ls_false_rescale = rotate(ls_false_rescale, angle)

    left = 124
    right = 1345

    top = 122
    bottom = 1378

    ls_false_rescale = ls_false_rescale[top:bottom,left:right]

    return ls_false_rescale


def bar_plot_with_colors(hist,colors,labels):

    # Matplotlib takes an RGB *fraction* as input for colors
    cm = [tuple(1.*np.array(c)/255.) for c in colors] # Matplotlib colormap takes a fraction

    plt.figure(figsize=(10, 3))
    plt.title('Fractional area of color clusters')
    plt.subplot(121)
    plt.bar(labels,hist,color=cm)
