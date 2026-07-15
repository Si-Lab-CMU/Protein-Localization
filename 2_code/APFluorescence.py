'''
APFluorescence contains functions related to fluorescence intensity analysis.

Created 20151111 by jt, original code by Yonggun
'''

# import modules
import numpy as np
import kmeans1d
import cv2 # openCV
from skimage.util import img_as_ubyte
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt

################################################################################
### functions
def fluor_intensity(img, imgfluo, contours, params):
    '''fluor_intensity measures average and total fluorescence intensity
    using a contour as a boundary for the cell.

    Parameters
    ----------
    img : 2D np.array
         phase contrast or bright field image
    imgfluo : 2D np.array
        fluorescence image
    contours
        list of contours
    params : dict
         dictionary with parameters from .yaml

    Returns
    -------
    contours
        list of contours
    avg_int_cells : np.array
        numpy array with average intensity per cell
    tot_int_cells : np.array
        numpy array with total intensity per cell
    mean_background : int
        mean background fluorecence over the image
    '''

    avg_int_cells = [] # numpy array with average intensity per cell
    tot_int_cells = [] # numpy array with total intensity per cell

    # ------------------------------------------------------------------
    # Huijing
    # find background intensities of the whole image
    # Using 1D k-means clustering
    # ------------------------------------------------------------------
    # kmeans1d: Globally Optimal Efficient 1D k‑means Clustering
    # https://www.dannyadam.com/blog/2019/07/kmeans1d-globally-optimal-efficient-1d-k-means/
    # ------------------------------------------------------------------
    # The 1d pixel intensities are first classfied into background pixels and fluorescent pixels
    # The centroid of the background pixels is used as the background mean
    # ------------------------------------------------------------------
    
    imgfluo_flat = imgfluo.flatten()
    clusters, centroids = kmeans1d.cluster(imgfluo_flat, 2)
    centroids = sorted(centroids)
    mean_background = centroids[0]
    # print(centroids)
    
    for h, cnt in reversed(list(enumerate(contours))):
        # find intensity of the cell
        int_mask = np.zeros(img.shape, np.uint8)

        # contour is a demarcation of the cell
        # this changes the mask to just the cell of interest, [cnt]
        cv2.drawContours(int_mask,[cnt],0,255,-1)
#        cv2.drawContours(int_mask,[cnt],0,255,10) #increase cell contour thickness #FS20200513

        # ---------------------
        # plt.imshow(int_mask)
        # plt.colorbar()
        # plt.show()
        
        # plt.imshow(imgfluo)
        # plt.colorbar()
        # plt.show()
        
        # plt.cla()
        # ---------------------

        # average intensity
        # average intensity by pixels ### Note, this is not average intensity by volumn
        avg_int = cv2.mean(imgfluo, mask = int_mask)

        # total intensity
        area = cv2.contourArea(cnt) # area of the cell based on contour
#        tot_int = avg_int[0] * area # is just avgerage times area
        tot_int = avg_int[0] * area - mean_background * area # is just avgerage times area #FS20200416 background subtraction

        # append data
        avg_int_cells = np.append(avg_int_cells, avg_int[0])
        tot_int_cells = np.append(tot_int_cells, tot_int)

    return contours, avg_int_cells, tot_int_cells, mean_background

    # ------------------------------------------------------------------
    
    """
    background_mask = np.zeros(img.shape, np.uint8)
    background_mask = 255 - background_mask
    mean_background = cv2.mean(imgfluo, mask = background_mask)

    for h, cnt in reversed(list(enumerate(contours))):
        # find intensity of the cell
        int_mask = np.zeros(img.shape, np.uint8)

        # contour is a demarcation of the cell
        # this changes the mask to just the cell of interest, [cnt]
        cv2.drawContours(int_mask,[cnt],0,255,-1)
#        cv2.drawContours(int_mask,[cnt],0,255,10) #increase cell contour thickness #FS20200513
        cv2.drawContours(background_mask,[cnt],0,255,-1)

        # average intensity
        avg_int = cv2.mean(imgfluo, mask = int_mask)

        # total intensity
        area = cv2.contourArea(cnt) # area of the cell based on contour
#        tot_int = avg_int[0] * area # is just avgerage times area
        tot_int = avg_int[0] * area - mean_background[0] * area # is just avgerage times area #FS20200416 background subtraction

        # append data
        avg_int_cells = np.append(avg_int_cells, avg_int[0])
        tot_int_cells = np.append(tot_int_cells, tot_int)

    return contours, avg_int_cells, tot_int_cells, mean_background[0]
    
    """

