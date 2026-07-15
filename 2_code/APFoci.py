'''
APFoci contains functions for foci analysis for agarPad

Created 20151210 by jt, original code by Yonggun/Fangwei
Edited 20151217 by jt, added laplacian/difference of gaussian method
'''

# import modules
import numpy as np
from numpy import unravel_index
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cv2 # openCV
from scipy import ndimage, optimize
from scipy.optimize import leastsq
import scipy.ndimage.filters as filters
from skimage.feature import blob_dog, blob_log

# agarPad modlues
from APLibrary import contour_stats2, crop_cell_box

################################################################################
### functions
def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments
    width_x and width_y are 2*sigma x and sigma y of the guassian
    """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit
    if params are not provided, they are calculated from the moments
    params should be (height, x, y, width_x, width_y)"""
    gparams = moments(data) # create guess parameters.
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -data)
    p, success = optimize.leastsq(errorfunction, gparams)
    return p

def foci_thr(img, img_foci, contours, cell_indicies, params):
    '''foci_thr finds foci based on a theshold method and then fits a 2D
    Gaussian. The returned information are the parameters of this Gaussian.
    All the information is returned in the form of np.arrays which are the
    length of the number of found foci across all cells in the image.
    The cell index indicates when there are multiple foci per cell.

    Parameters
    ----------
    img : 2D np.array
        phase contrast or bright field image
    img_foci : 2D np.array
        fluorescent image with foci hopefully
    contours
        list of contours
    cell_indicies : 1D np.array
        list of unique numbers for the contours (cells)
    params : dict
         dictionary with parameters from .yaml

    Returns
    -------
    indicies : 1D np.array
        list of indicies of cells
    disp_l : 1D np.array
        displacement on long axis, in um, of a foci from the center of the cell
    disp_w : 1D np.array
        displacement on short axis, in um, of a foci from the center of the cell
    foci_wx : 1D np.array
        width of foxi in long axis direction in pixels
    foci_wy : 1D np.array
        width of foci in sort axis direction in pixels
    foci_h : 1D np.array
        height of foci (intensity value of fluorecent image at Gaussian peak)
    '''

    # declare parameters for foci finding
    threshold = params['foci_threshold'][0]
    neighborhood_size = params['foci_neighborhood'][0]

    # declare arrays which will hold foci data
    indicies = [] # index of the cell
    disp_l = [] # displacement in length of foci from cell center
    disp_w = [] # displacement in width of foci from cell center
    foci_wx = [] # foci x width (from 2D gaussian fit)
    foci_wy = [] # foci y width (from 2D gaussian fit)
    foci_h1 = [] # foci peak height (from 2D gaussian fit)
    foci_h2 = [] # foci total amount (from 2D gaussian fit)
    foci_h3 = [] # foci peak height (from raw image)
    foci_h4 = [] # foci total amount (from raw image)

    # go through contours and look for foci
    for h, cnt in enumerate(contours):
        # declare arrays which hold information for this cell
        c_index = cell_indicies[h] # index number for this cell

        # calculate contour information
        rect, box, length, width, angle, area = contour_stats2(cnt)

        # make bigger box around cells
        sz_img = length+30
        sz_lth = length+25
        sz_wth = width+10
        s_piece = img_foci[np.int0(rect[0][1]-sz_img/2):np.int0(rect[0][1]+sz_img/2), np.int0(rect[0][0]-sz_img/2):np.int0(rect[0][0]+sz_img/2)]
        s_piece_pc = img[np.int0(rect[0][1]-sz_img/2):np.int0(rect[0][1]+sz_img/2), np.int0(rect[0][0]-sz_img/2):np.int0(rect[0][0]+sz_img/2)]

        # rotate this bigger box so cell is horizontal
        rows, cols = s_piece.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        try:
            b = cv2.warpAffine(s_piece,M,(cols,rows))
            b_pc = cv2.warpAffine(s_piece_pc,M,(cols,rows))
        except:
            print('Affine transformation failed')
            continue

        # cut the image again to something smaller for foci detection
        sz_imgR = b.shape[0]
        c = b[sz_imgR/2-sz_wth/2:sz_imgR/2+sz_wth/2, sz_imgR/2-sz_lth/2:sz_imgR/2+sz_lth/2]
        c_pc = b_pc[sz_imgR/2-sz_wth/2:sz_imgR/2+sz_wth/2, sz_imgR/2-sz_lth/2:sz_imgR/2+sz_lth/2]

        # find 'center of cell' relative to the box
        # this would be better if replaced with a centroid locator or foci were detected as distance from 'spine' of cell
        # find centroid of cell
        c_center_lth = c.shape[1] / float(2)
        c_center_wth = c.shape[0] / float(2)

        # blur the foci image a little before looking for foci
        cc = cv2.GaussianBlur(c,(3,3),0)

        # plot the progression of the image transformations
        if False: #params['debug_foci']:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(4,2,1)
            plt.imshow(s_piece, cmap='gray')
            ax = fig.add_subplot(4,2,3)
            plt.imshow(b, cmap='gray')
            ax = fig.add_subplot(4,2,5)
            plt.imshow(c, cmap='gray')
            ax = fig.add_subplot(4,2,7)
            plt.imshow(cc, cmap='gray')
            ax = fig.add_subplot(4,2,2)
            plt.imshow(s_piece_pc, cmap='gray')
            ax = fig.add_subplot(4,2,4)
            plt.imshow(b_pc, cmap='gray')
            ax = fig.add_subplot(4,2,6)
            plt.imshow(c_pc, cmap='gray')
            plt.show(block=False)

        # filter to find maxima, which are possible foci
        data_max = filters.maximum_filter(cc, neighborhood_size)
        maxima = (cc == data_max)
        # fig = plt.figure(figsize=(10,10))
        # plt.imshow(maxima)
        # plt.show()
        data_min = filters.minimum_filter(cc, neighborhood_size)
        # the foci should be a certain threshold above the minima
        try: # I don't even think this does anything
            diff = ((data_max / data_min) > threshold)
        except:
            ('Nothing above foci treshold')
            continue
        maxima[diff == 0] = 0

        # zero out areas not in cell area
        sz_imgC = cc.shape
        maxima[0:sz_imgC[0]/2-width/2,:] = 0 # top
        maxima[sz_imgC[0]/2+width/2:sz_imgC[0],:] = 0 # bottom
        maxima[:,0:sz_imgC[1]/2-length/2] = 0 # left
        maxima[:,sz_imgC[1]/2+length/2:sz_imgC[1]] = 0 # right

        # make labels for the maxima
        labeled, num_objects = ndimage.label(maxima)

        # this finds the location of these labels
        slices = ndimage.find_objects(labeled)

        # these will hold information abou foci position temporarily
        x, y = [], []
        xx, yy, xxw, yyw = [], [], [], []
        sz_fit = 6 # increase the size around the foci for gaussian fitting

        # loop through each potenial foci
        for dy, dx in slices:
            x_center = (dx.start + dx.stop - 1)/2
            x.append(x_center) # for plotting
            y_center = (dy.start + dy.stop - 1)/2
            y.append(y_center) # for plotting

            if x_center<sz_fit+2 or y_center<sz_fit+2:
#                print('throw out 1 (this should not trigger)')
                continue

            # increase the size around the supposed foci to use for gaus fit
            xl = x_center-sz_fit
            xr = x_center+sz_fit+1
            yt = y_center-sz_fit
            yb = y_center+sz_fit+1
            if xl<=0 or xr<=0 or yt<=0 or yb<=0:
#                print('throw out 2 (this should not trigger)')
                continue

            # cut out a small image from the blurred foci image
            data = cc[yt:yb, xl:xr]
            data_0=c[yt:yb, xl:xr]

            rdd, cdd = data.shape
            if rdd < 2*sz_fit+1 or cdd < 2*sz_fit+1:
#                print('throw out 3 (this should not trigger)')
                continue

            # fit gaussian to proposed foci in small box
            p = fitgaussian(data)
            (peak, xc, yc, width_x, width_y) = p

            # filter out huge foci or foci outside the box
            # I don't think it's possible for foci to be this big, as the box ir pretty small, whatever.
            if width_x > 17 or width_y > 17:
                if params['debug_foci']:
                    print('foci is very large')
                continue
            elif xc<0 or xc>rdd:
                print('throw out 5 (this should not trigger)')
                continue
            elif yc<0 or yc>cdd:
                print('throw out 6 (this should not trigger)')
                continue
            elif peak<0:
                print('throw out 7 (foci peak negative)')
                continue
            else:
                # find x an y position
                xxx = x_center - sz_fit + xc
                yyy = y_center - sz_fit + yc
                xx = np.append(xx, xxx) # for plotting
                yy = np.append(yy, yyy) # for plotting
                xxw = np.append(xxw, width_x) # for plotting
                yyw = np.append(yyw, width_y) # for plotting

                # calculate distance of foci from middle of cell
                disp_x = (xxx - c_center_lth)
                disp_y = (yyy - c_center_wth)

                # append foci information to the list
                indicies = np.append(indicies, c_index) # index of the cell
                disp_l = np.append(disp_l, disp_x*params['pxl2um'])
                disp_w = np.append(disp_w, disp_y*params['pxl2um'])
                foci_wx = np.append(foci_wx, width_x)
                foci_wy = np.append(foci_wy, width_y)
                foci_h1 = np.append(foci_h1, peak)
                foci_h2 = np.append(foci_h2, peak*np.pi*width_x*width_y)
                foci_h3 = np.append(foci_h3, np.amax(data_0))
                foci_h4 = np.append(foci_h4, np.sum(data_0))
                

                if params['debug_foci']:
                    print(xxx, c_center_lth, disp_x, width_x, peak)
                    print(yyy, c_center_wth, disp_y, width_y, peak)

        # draw foci on image for quality control
        if params['debug_foci']:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(2,1,1)
            foci_overlay = (c*255).copy()
            plt.imshow(c, interpolation='nearest', cmap='gray')
            # add circles for where the maxima were
            for i, max_spot in enumerate(x):
                foci_center = Ellipse([x[i],y[i]],1,1,color=(0, 1.0, 1.0), linewidth=2, alpha=0.5)
                ax.add_patch(foci_center)
            # show the shape of the gaussian for recorded foci
            for i, spot in enumerate(xx):
                foci_ellipse = Ellipse([xx[i],yy[i]], xxw[i], yyw[i],color=(0, 1.0, 0.0), linewidth=2, fill=False, alpha=0.5)
                ax.add_patch(foci_ellipse)
            plt.title('blue = maxima, green = recorded foci')
            ax = fig.add_subplot(2,1,2)
            plt.imshow(c_pc, cmap='gray')
            plt.show()
            print('\n') # creates a space for foci data on terminal

    return indicies, disp_l, disp_w, foci_wx, foci_wy, foci_h1, foci_h2, foci_h3, foci_h4

def foci_lap(img, img_foci, contours, cell_indicies, params):
    '''foci_dog finds foci using a laplacian convolution then fits a 2D
    Gaussian.

    The returned information are the parameters of this Gaussian.
    All the information is returned in the form of np.arrays which are the
    length of the number of found foci across all cells in the image.

    Parameters
    ----------
    img : 2D np.array
        phase contrast or bright field image
    img_foci : 2D np.array
        fluorescent image with foci hopefully
    contours
        list of contours
    cell_indicies : 1D np.array
        list of unique numbers for the contours (cells)
    params : dict
         dictionary with parameters from .yaml

    Returns
    -------
    indicies : 1D np.array
        list of indicies of cells
    disp_l : 1D np.array
        displacement on long axis, in um, of a foci from the center of the cell
    disp_w : 1D np.array
        displacement on short axis, in um, of a foci from the center of the cell
    foci_wx : 1D np.array
        width of foxi in long axis direction in pixels
    foci_wy : 1D np.array
        width of foci in sort axis direction in pixels
    foci_h : 1D np.array
        height of foci (intensity value of fluorecent image at Gaussian peak)
    '''

    # declare arrays which will hold foci data
    indicies = [] # index of the cell
    disp_l = [] # displacement in length of foci from cell center
    disp_w = [] # displacement in width of foci from cell center
    foci_wx = [] # foci x width (from 2D gaussian fit)
    foci_wy = [] # foci y width (from 2D gaussian fit)
    foci_h1 = [] # foci peak height (from 2D gaussian fit)
    foci_h2 = [] # foci total amount (from 2D gaussian fit)
    foci_h3 = [] # foci peak height (from raw image)
    foci_h4 = [] # foci total amount (from raw image)

    # define parameters for foci finding
    minsig = params['foci_log_minsig']
    maxsig = params['foci_log_maxsig']
    thresh = params['foci_log_thresh']
    peak_med_ratio = params['foci_log_peak_med_ratio']

    # blur image for subtraction
    #img_foci = img_foci.astype('uint8')
    #img_foci_gaus = cv2.GaussianBlur(img_foci,(21,21),0)
    img_foci_med_blur = cv2.medianBlur(img_foci,5)

    # go through contours and look for foci
    for h, cnt in enumerate(contours):
        # declare arrays which hold information for this cell
        c_index = cell_indicies[h] # index number for this cell

## YG's code
        # calculate contour information
        rect, box, length, width, angle, area = contour_stats2(cnt)

        # make bigger box around cells
        sz_img = length+30
        sz_lth = length+25
        sz_wth = width+10
        s_piece = img_foci[np.int0(rect[0][1]-sz_img/2):np.int0(rect[0][1]+sz_img/2), np.int0(rect[0][0]-sz_img/2):np.int0(rect[0][0]+sz_img/2)]
        s_piece_pc = img[np.int0(rect[0][1]-sz_img/2):np.int0(rect[0][1]+sz_img/2), np.int0(rect[0][0]-sz_img/2):np.int0(rect[0][0]+sz_img/2)]

        # rotate this bigger box so cell is horizontal
        rows, cols = s_piece.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        try:
            b = cv2.warpAffine(s_piece,M,(cols,rows))
            b_pc = cv2.warpAffine(s_piece_pc,M,(cols,rows))
        except:
            print('Affine transformation failed')
            continue

        # cut the image again to something smaller for foci detection
        sz_imgR = b.shape[0]
        c = b[ int(sz_imgR/2-sz_wth/2) : int(sz_imgR/2+sz_wth/2), int(sz_imgR/2-sz_lth/2) : int(sz_imgR/2+sz_lth/2)]
        c_pc = b_pc[ int(sz_imgR/2-sz_wth/2) : int(sz_imgR/2+sz_wth/2), int(sz_imgR/2-sz_lth/2) : int(sz_imgR/2+sz_lth/2)]


## JT's code
#        # calculate contour information
#        rect, box, length, width, angle, area = contour_stats2(cnt)
#
        # calculate median cell intensity. Used to filter foci
        int_mask = np.zeros(img_foci.shape, np.uint8)
        cv2.drawContours(int_mask,[cnt],0,255,-1)
        avg_int = cv2.mean(img_foci, mask = int_mask)
        avg_int = avg_int[0]
#
#        c = crop_cell_box(img_foci, cnt, 5, 5, params)
#        c_pc = crop_cell_box(img, cnt, 5, 5, params)
#
        # find 'center of cell' relative to the box
        # this would be better if replaced with a centroid locator or foci were detected as distance from 'spine' of cell
        c_center_lth = c.shape[1] / float(2)
        c_center_wth = c.shape[0] / float(2)

        # transform image before foci detection?
        cc = c
        sz_imgC = cc.shape


        # skip if contour is NaN
        if sz_imgC[0]*sz_imgC[1]==0:
            continue
        
        #c_blur_gaus = crop_cell_box(img_foci_gaus, cnt, 5, 5, params)
        # blur by width of cell
        blur_size = int(width)
        if blur_size % 2 == 0:
            blur_size += 1
        c_blur_gaus = cv2.GaussianBlur(c,(blur_size,blur_size),0)
        # subtract background
        c_subtract_gaus = cc - c_blur_gaus
        c_subtract_gaus[c_subtract_gaus > 10000] = 0

        # this is not used, gaussian blur is better
        # c_blur_med = crop_cell_box(img_foci_med_blur, cnt, 5, 5, params)
        # c_subtract_med = cc - c_blur_med
        # c_subtract_med[c_subtract_med > 10000] = 0

        # plot the progression of the image transformations
        # this has been moved to save with the debug image
        if False:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(3,2,1)
            plt.add_title('fluor image')
            plt.imshow(c, interpolation='nearest', cmap='gray')
            ax = fig.add_subplot(3,2,2)
            plt.add_title('phase image')
            plt.imshow(c_pc, interpolation='nearest', cmap='gray')
            ax = fig.add_subplot(3,2,3)
            plt.add_title('gaussian blur')
            plt.imshow(c_blur_gaus, interpolation='nearest', cmap='gray')
            ax = fig.add_subplot(3,2,4)
            plt.add_title('median blur')
            plt.imshow(c_blur_med, interpolation='nearest', cmap='gray')
            ax = fig.add_subplot(3,2,5)
            plt.add_title('gaussian subtraction')
            plt.imshow(c_subtract_gaus, interpolation='nearest', cmap='gray')
            ax = fig.add_subplot(3,2,6)
            plt.add_title('median subtraction')
            plt.imshow(c_subtract_med, interpolation='nearest', cmap='gray')
            plt.show()

        # find blobs using difference of gaussian
        over_lap = .95 # if two blobs overlap by more than this fraction, smaller blob is cut
        numsig = maxsig - minsig + 1 # number of division to consider (height of z cube) set this heigh so it considers all pixels
        blobs = blob_log(c_subtract_gaus, min_sigma=minsig, max_sigma=maxsig, overlap=over_lap, num_sigma=numsig, threshold=thresh)

        # these will hold information abou foci position temporarily
        x, y, r = [], [], []
        xx, yy, xxw, yyw = [], [], [], []
        gfitmins = []
        gfitmaxs = []

        # loop through each potenial foci
        for blob in blobs:
            yloc, xloc, sig = blob # x locachrtion, y location, and sigma of gaus
            radius = np.ceil(np.sqrt(2)*sig)
            x.append(xloc) # for plotting
            y.append(yloc) # for plotting
            r.append(radius)

            sz_fit = radius # increase the size around the foci for gaussian fitting

            #remove blob if not in cell box
            if (xloc < sz_imgC[1]/2-length/2 or xloc > sz_imgC[1]/2+length/2 or
                yloc < sz_imgC[0]/2-width/2 or yloc > sz_imgC[0]/2+width/2):
                if params['debug_foci']: print('blob not in cell area')
                continue

            # cut out a small image from origincal image to fit gaussian
            gfit_area = cc[yloc-sz_fit:yloc+sz_fit, xloc-sz_fit:xloc+sz_fit]
            gfit_rows, gfit_cols = gfit_area.shape

            #print('gfit_area', gfit_rows, gfit_cols)
            
#            gfit_area_0 = c[yloc-sz_fit:yloc+sz_fit, xloc-sz_fit:xloc+sz_fit]
            gfit_area_0 = c[max(0,yloc-1*sz_fit):min(c.shape[0],yloc+1*sz_fit), max(0,xloc-1*sz_fit):min(c.shape[1],xloc+1*sz_fit)]

            # fit gaussian to proposed foci in small box
            p = fitgaussian(gfit_area)
            (peak, xc, yc, width_x, width_y) = p
            gfit_median = np.median(gfit_area)
            gfit_max = np.max(gfit_area)
            gfit_area = np.mean(gfit_area)

            # plt.figure(figsize=(4,4))
            # plt.imshow(gfit_area, interpolation='nearest', cmap='gray')
            # plt.show()

            if params['debug_foci']:
                print(np.min(gfit_area), np.max(gfit_area), gfit_median, avg_int, peak)

            # filter out foci based size, location, and shape
            # if width_x > params['foci_log_maxsig'] or width_y > radius*4:
            #     if params['debug_foci']: print('foci is very large')
            #     continue
            if xc <= 0 or xc >= gfit_cols or yc <= 0 or yc >= gfit_rows:
                if params['debug_foci']: print('throw out foci (gaus fit not in gfit_area)')
                continue
            elif peak/avg_int < peak_med_ratio:
                if params['debug_foci']: print('peak does not pass height test')
                continue
            else:
                # find x an y position
                xxx = xloc - sz_fit + xc
                yyy = yloc - sz_fit + yc
                xx = np.append(xx, xxx) # for plotting
                yy = np.append(yy, yyy) # for plotting
                xxw = np.append(xxw, width_x) # for plotting
                yyw = np.append(yyw, width_y) # for plotting

                # calculate distance of foci from middle of cell
                disp_x = (xxx - c_center_lth)
                disp_y = (yyy - c_center_wth)

                # append foci information to the list
                indicies = np.append(indicies, c_index) # index of the cell
                disp_l = np.append(disp_l, disp_x*params['pxl2um'])
                disp_w = np.append(disp_w, disp_y*params['pxl2um'])
                foci_wx = np.append(foci_wx, width_x)
                foci_wy = np.append(foci_wy, width_y)
                foci_h1 = np.append(foci_h1, peak)
                foci_h2 = np.append(foci_h2, peak*np.pi*width_x*width_y)
                foci_h3 = np.append(foci_h3, np.amax(gfit_area_0))
                foci_h4 = np.append(foci_h4, np.sum(gfit_area_0))

                if params['debug_foci']:
                    print(disp_x, width_x)
                    print(disp_y, width_y)

        # draw foci on image for quality control
        if params['debug_foci']:
            # processing of image
            fig = plt.figure(figsize=(12,12))
            ax = fig.add_subplot(3,2,1)
            plt.title('fluor image')
            plt.imshow(c, interpolation='nearest', cmap='gray')
            ax = fig.add_subplot(3,2,2)
            plt.title('phase image')
            plt.imshow(c_pc, interpolation='nearest', cmap='gray')
            ax = fig.add_subplot(3,2,3)
            plt.title('gaussian blur')
            plt.imshow(c_blur_gaus, interpolation='nearest', cmap='gray')
            ax = fig.add_subplot(3,2,5)
            plt.title('gaussian subtraction')
            plt.imshow(c_subtract_gaus, interpolation='nearest', cmap='gray')


            ax = fig.add_subplot(3,2,4)
            plt.title('DoG blobs')
            plt.imshow(c_subtract_gaus, interpolation='nearest', cmap='gray')
            # add circles for where the blobs are
            for i, max_spot in enumerate(x):
                foci_center = Ellipse([x[i],y[i]],r[i],r[i],color=(1.0, 1.0, 0), linewidth=2, fill=False, alpha=0.5)
                ax.add_patch(foci_center)

            # show the shape of the gaussian for recorded foci
            ax = fig.add_subplot(3,2,6)
            plt.title('final foci')
            plt.imshow(c, interpolation='nearest', cmap='gray')
            # print foci that pass and had gaussians fit
            for i, spot in enumerate(xx):
                foci_ellipse = Ellipse([xx[i],yy[i]], xxw[i], yyw[i],color=(0, 1.0, 0.0), linewidth=2, fill=False, alpha=0.5)
                ax.add_patch(foci_ellipse)
            plt.show()
#            print('\n') # creates a space for foci data on terminal

#            img_title = params['path_debug'] + 'foci_lap_cell_' + str(c_index) + '.jpg'
#            fig.savefig(img_title)
#            plt.close(fig)

    return indicies, disp_l, disp_w, foci_wx, foci_wy, foci_h1, foci_h2, foci_h3, foci_h4
