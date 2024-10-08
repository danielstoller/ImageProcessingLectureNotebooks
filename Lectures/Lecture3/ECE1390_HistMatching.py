 
'''
Welcome to the Histogram Matching Program!
 
Given a source image and a reference image, this program
returns a modified version of the source image that matches
the histogram of the reference image.
 
Image Requirements:
  - The sizes of the source image and reference image do not
    have to be the same.
  - Both images have to be in the same format Color or Gray
 
Usage:
  histogram_matching <source_image> <ref_image>

Modified from
Project: Histogram Matching Using OpenCV
Author: Addison Sears-Collins
Date created: 9/27/2019
Python version: 3.7

https://automaticaddison.com/how-to-do-histogram-matching-using-opencv/
'''
 
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import matplotlib.pyplot as plt # Import matplotlib functionality
import math
 
 
def calculate_cdf(histogram):
    """
    This method calculates the cumulative distribution function
    :param array histogram: The values of the histogram
    :return: normalized_cdf: The normalized cumulative distribution function
    :rtype: array
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()
 
    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())
 
    return normalized_cdf
 
def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        lookup_val
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table
 
def match_histograms(src_image, ref_image):
    """
    This method matches the source image histogram to the
    reference signal
    :param image src_image: The original source image
    :param image  ref_image: The reference image
    :return: image_after_matching
    :rtype: image (array)
    """

    if ( len(src_image.shape)==3):
        # Split the images into the different color channels
        # b means blue, g means green and r means red
        src_b, src_g, src_r = cv2.split(src_image)
        ref_b, ref_g, ref_r = cv2.split(ref_image)
    
        # Compute the b, g, and r histograms separately
        # The flatten() Numpy method returns a copy of the array c
        # collapsed into one dimension.
        src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0,256])
        src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0,256])
        src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0,256])    
        ref_hist_blue, bin_3 = np.histogram(ref_b.flatten(), 256, [0,256])    
        ref_hist_green, bin_4 = np.histogram(ref_g.flatten(), 256, [0,256])
        ref_hist_red, bin_5 = np.histogram(ref_r.flatten(), 256, [0,256])
    
        # Compute the normalized cdf for the source and reference image
        src_cdf_blue = calculate_cdf(src_hist_blue)
        src_cdf_green = calculate_cdf(src_hist_green)
        src_cdf_red = calculate_cdf(src_hist_red)
        ref_cdf_blue = calculate_cdf(ref_hist_blue)
        ref_cdf_green = calculate_cdf(ref_hist_green)
        ref_cdf_red = calculate_cdf(ref_hist_red)
    
        # Make a separate lookup table for each color
        blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
        green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
        red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)
    
        # Use the lookup function to transform the colors of the original
        # source image
        blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
        green_after_transform = cv2.LUT(src_g, green_lookup_table)
        red_after_transform = cv2.LUT(src_r, red_lookup_table)
    
        # Put the image back together
        image_after_matching = cv2.merge([
            blue_after_transform, green_after_transform, red_after_transform])
        image_after_matching = cv2.convertScaleAbs(image_after_matching)
    else:
        # Handle gray scale
        
        # The flatten() Numpy method returns a copy of the array c
        # collapsed into one dimension.
        src_hist, bin_0 = np.histogram(src_image.flatten(), 256, [0,256])
        ref_hist, bin_3 = np.histogram(ref_image.flatten(), 256, [0,256])    
        
        # Compute the normalized cdf for the source and reference image
        src_cdf = calculate_cdf(src_hist)
        ref_cdf = calculate_cdf(ref_hist)
    
        # Make a separate lookup table for each color
        lookup_table = calculate_lookup(src_cdf, ref_cdf)
        
        # Use the lookup function to transform the colors of the original
        # source image
        image_after_matching = cv2.LUT(src_image, lookup_table)
        image_after_matching = cv2.convertScaleAbs(image_after_matching)

    return image_after_matching
 


def gamma_correction(image,gamma):
# This function does gamma correction
     
    invGamma = 1.0 / gamma
    
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
	
    # apply gamma correction using the lookup table
	
    return cv2.LUT(image, table)


def alpha_blend(img1,img2,alpha=math.nan):
    # This function does alpha blending of two images
    # if the third argument (alpha) is not defined then
    # use the alpha channels in the images to merge
    # otherwise use the value of alpha to set the whole image

    

    if(not math.isnan(alpha)):
       return cv2.addWeighted(img1, 1-alpha , img2, alpha, 0)
    else:
        imgA=img1.copy()
        imgB=img2.copy()

        if(len(imgA.shape)==3 & imgA.shape[2]==3):
            imgA = cv2.cvtColor(imgA,cv2.COLOR_RGB2RGBA)
        if(len(imgB.shape)==3 & imgB.shape[2]==3):
            imgB = cv2.cvtColor(imgB,cv2.COLOR_RGB2RGBA)

        dtype=imgA.dtype

        imgA=imgA.astype('float32',casting='unsafe')     
        imgB=imgB.astype('float32',casting='unsafe')     

        out = np.zeros([imgA.shape[0], imgA.shape[1], 3])
              
        out[:,:,0]=imgA[:,:,0]*imgA[:,:,3]/255+imgB[:,:,0]*(1-imgA[:,:,3]/255)*imgB[:,:,3]/255
        out[:,:,1]=imgA[:,:,1]*imgA[:,:,3]/255+imgB[:,:,1]*(1-imgA[:,:,3]/255)*imgB[:,:,3]/255
        out[:,:,2]=imgA[:,:,2]*imgA[:,:,3]/255+imgB[:,:,2]*(1-imgA[:,:,3]/255)*imgB[:,:,3]/255
        
        return out.astype(dtype,casting='unsafe')     


                










    