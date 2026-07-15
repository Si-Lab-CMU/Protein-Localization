'''
APPhase contains functions related to phase contrast (size) analysis.

Created 20151111 by jt, original code by Yonggun
'''

# import modules
import numpy as np
import cv2 # openCV
from skimage.util import img_as_ubyte
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt

# agarPad modules
from APLibrary import plot_contours, contour_stats1, contour_stats2

################################################################################
### functions
def pc_profile_sizing(img, contours, params):
    '''pc_profile_sizing finds the length and width of cells based on a
    thresholding method of the length and width profiles. It will filter
    contours out based on length and width parameters

    Parameters
    ----------
    img : 2D np.array
         phase contrast or bright field image
    contours
        list of contours
    params : dict
         dictionary with parameters from .yaml

    Returns
    -------
    contours
        list of contours
    lth_pc : np.array
        list of profile lengths in pixels
    wth_pc : np.array
        list of profile widths in pixels
    '''

    # define some variables from the global parameter dictionary
    pc_int_thr = params['pc_int_thr']
    pxl2um = params['pxl2um']

    lth_pc = [] # length of cell
    wth_pc = [] # width of cell pc

    for h, cnt in reversed(list(enumerate(contours))):

        # calculate contour information
        rect, box, length, width, angle, area = contour_stats2(cnt)

        # make a larger box around the contour
        sz_img = length + 30
        sz_lth = length + 25
        sz_wth = width + 10

        # crop out a rectangle around the cell
        s_piece_pc = img[np.int0(rect[0][1]-sz_img/2):np.int0(rect[0][1]+sz_img/2), np.int0(rect[0][0]-sz_img/2):np.int0(rect[0][0]+sz_img/2)]
        # rotate image so cell is horizontal
        b_pc = ndimage.rotate(s_piece_pc, angle)

        ####################################################################
        ### determine the length of cell

        # find center line of cell.
        sz_imgR = b_pc.shape[0]
        #print("hahaha", sz_imgR/2-3,sz_imgR/2+3, sz_imgR/2-sz_lth/2, sz_imgR/2+sz_lth/2)
        d_pc = b_pc[int(sz_imgR/2-3) : int(sz_imgR/2+3), 
                    int(sz_imgR/2-sz_lth/2) : int(sz_imgR/2+sz_lth/2)]
        kkk_pc = np.mean(d_pc,0) #profile of 6px

        # make sure you actually got something here
        len_kkk = len(kkk_pc)
        if (len_kkk)==0:
            del contours[h]
            continue

        # x position in profile
        xxx = np.arange(0, len_kkk)

        # find min and max intesity values
        mx_pc = max(kkk_pc)
        mn_pc = min(kkk_pc)

        # try to do bg subtraction and normalization
        try:
            kkk_pc_r=1-(kkk_pc-mn_pc)/(mx_pc-mn_pc)
        except:
            del contours[h]
            continue

        # find mean max of center of length
        mx_avg = np.mean(kkk_pc_r[ int(len_kkk/2-10) : int(len_kkk/2+11) ])
        mx_avg = mx_avg*pc_int_thr
        # line with value of that avg
        avg = np.zeros((len_kkk,1))
        avg[:] = mx_avg

        ### Determine the length based on PC image
        # left side threshold point
        l_th = find_threshold(kkk_pc_r, pc_int_thr)

        # right side threshold point
        # flip the profile to look from the other side
        kkk_pc_r_reversed = kkk_pc_r[::-1]
        r_th = find_threshold(kkk_pc_r_reversed, pc_int_thr)
        # the point must be subtracted from other side
        r_th = len_kkk-r_th-1

        # filter out cells if the length makes no sense
        if np.isnan(r_th) or np.isnan(l_th) or np.isnan(r_th-l_th) or r_th==0 or l_th==0:
            del contours[h]
            continue
        else:
            ll = r_th - l_th # length

        # save variables for plotting later
        if params['debug_phase']:
            xxx_l = xxx
            kkk_pc_r_l = kkk_pc_r
            avg_l = avg
            l_th_l = l_th
            mx_avg_l = mx_avg
            r_th_l = r_th

        ####################################################################
        ### determine the width of cell
        wd = width-2

        # split width into two parts to avoid contricted centers
        try:
            ll_b=int(l_th+wd/2)
            lr_b=int((l_th+r_th)/2-wd/2)
            rl_b=int((l_th+r_th)/2+wd/2)
            rr_b=int(r_th-wd/2)
        except:
            del contours[h]
            continue

        f_pc = b_pc[ int(sz_imgR/2-sz_wth/2-3) : int(sz_imgR/2+sz_wth/2+5), 
                     int(sz_imgR/2-sz_lth/2) : int(sz_imgR/2+sz_lth/2)]

        r,c = f_pc.shape
        if r <= 0 or c <= 0:
            del contours[h]
            continue

        l_img=f_pc[0:r, ll_b:lr_b]
        r_img=f_pc[0:r, rl_b:rr_b]
        if False:
            plt.imshow(l_img, cmap=plt.cm.gray, interpolation='nearest')
            plt.show()

        l_th_mean = 0
        r_th_mean = 0
        kkk_all = np.zeros(r)

        # Define the range to average the width
#        img_piece=f_pc[0:r, ll_b:rr_b]
        img_piece=np.concatenate((l_img, r_img),axis=1) #combine two parts that do not include septum FS20210714
        rr, cc = img_piece.shape
        llength = ll_b
        rlength = rr_b

        if False:
            plt.imshow(img_piece, cmap=plt.cm.gray, interpolation='nearest')
            plt.show()

        # Calculate the mean width across all middle sections from above
        for pxl in range(cc):
            try:
                kkk_pc_w=img_piece[0:r, pxl]
            except:
                del contours[h]
                continue
            #kkk_pc_w=np.mean(s_pc,1)
            len_kkk=len(kkk_pc_w)
            if (len_kkk)==0:
                del contours[h]
                continue

            xxx_w = np.arange(0, len_kkk)

            mx_pc = np.max(kkk_pc_w)
            mn_pc = np.min(kkk_pc_w)

            # normalize data
            try:
                kkk_pc_r = 1 - (kkk_pc_w-mn_pc) / (mx_pc-mn_pc)
            except:
                del contours[h]
                continue
            #mx_avg=np.mean(kkk_pc_r[len_kkk/2-1:len_kkk/2+2])
            mx_avg= np.max(kkk_pc_r[ int(len_kkk/2-1) : int(len_kkk/2+2)])
            avg= np.zeros((len_kkk,1))
            mx_avg= mx_avg*pc_int_thr
            avg[:]= pc_int_thr

            if False:
                plt.plot(xxx_w, kkk_pc_r, 'b-', xxx_w, avg, 'g-')
                plt.show()

            # Determine the width based on threshold
            # left side threshold point
            l_th = find_threshold(kkk_pc_r, pc_int_thr)

            # right side threshold point
            # flip the profile to look from the other side
            kkk_pc_r_reversed = kkk_pc_r[::-1]
            r_th = find_threshold(kkk_pc_r_reversed, pc_int_thr)
            # the point must be subtracted from other side
            r_th = len_kkk-r_th-1

            # update average on fly
            l_th_mean = (l_th_mean*pxl + l_th) / (pxl+1)
            r_th_mean = (r_th_mean*pxl + r_th) / (pxl+1)

        # calculate width as distance between means
        ww = r_th_mean - l_th_mean

        # filtering data once again
        if np.isnan(l_th_mean) or np.isnan(r_th_mean) or np.isnan(r_th_mean-l_th_mean) or l_th_mean==0 or r_th_mean==0:
            del contours[h]
            continue
        else:
            ww=r_th_mean-l_th_mean

            if np.isnan(ll) or np.isnan(ww):
                del contours[h]
                continue
            elif ll <= params['lth_cut_low']/pxl2um or \
              ww <= params['wth_cut_low']/pxl2um:
                del contours[h]
                continue
            elif ll > params['lth_cut_up']/pxl2um or \
              ww > params['wth_cut_up']/pxl2um:
                del contours[h]
                continue
            else:
                lth_pc=np.append(lth_pc, ll)
                wth_pc=np.append(wth_pc, ww)

        # plot the cell and associated profiles
        if params['debug_phase']:
            # draw contour on image mask for plotting
            contour_overlay = plot_contours(img, cnt)

            plt.figure(figsize=(16,16), dpi=80)

            plt.subplot(2,2,1)
            plt.title('Cell (or should be)')
            plt.xlabel('pixels')
            plt.ylabel('pixels')
            plt.imshow(b_pc, cmap=plt.cm.gray, interpolation='nearest')

            plt.subplot(2,2,3)
            plt.title('Length profile intensity')
            plt.plot(xxx_l, kkk_pc_r_l, 'b-', xxx_l, avg_l, 'g-', l_th_l, mx_avg_l, 'bo', r_th_l, mx_avg_l, 'bo')
            plt.xlabel('pixels')
            plt.ylabel('normalized intensity')

            plt.subplot(2,2,2)
            plt.title('Width profile intensity')
            plt.plot(xxx_w, kkk_pc_r, 'b-', xxx_w, avg, 'g-', l_th, pc_int_thr, 'bo', r_th, pc_int_thr, 'bo')
            plt.xlabel('pixels')
            plt.ylabel('normalized intensity')

            plt.subplot(2,2,4)
            plt.title('Location of cell in image')
            plt.imshow(contour_overlay, cmap=plt.cm.gray, interpolation='nearest')
            plt.xlabel('pixels')
            plt.ylabel('pixels')
            plt.show()

    return contours, lth_pc, wth_pc

def find_threshold(profile, thr):
    '''This travels along a profile and finds the point at which a certain threshold is met. It then uses a linear fit to find where the threhold was met with sub pixel resolution.

    Parameters
    ----------
    profile : np.array
        linear intensity profile
    thr : float
        threshold from 0 to 1 from which to find a point

    Returns
    -------
    thr_pt : int
        point along the profile that the threshold is met
    '''
    thr_pt = 0 # initialize position of where threhold is met
    n = 0 # holds position in profile array

    # loop over intensity profile
    for i in (profile):
        if i > thr: # look for first pixel value over threshold
            ax1 = n-1 # pixel array position before threshold met
            ay1 = profile[n-1] # intesity value at that position
            ax2 = n # pixxel array position after threshold met
            ay2 = i # intesity value at that threshold
            k = (ay2-ay1)/(ax2-ax1) # calculate slope
            # the slope should be postive and don't choose the first few points
            if k < 0 or n < 3:
                continue
            b = ay2 - k*ax2 # find intercept
            try:
                thr_pt = (thr - b) / k # find exact point thr met
            except:
                print('Threshold on profile not found.')
                continue
            break # break if this is found
        n += 1 # update position
    return thr_pt

# --------------------------------------------------------------------
# Huijing
# For phase contour, calc contour length and intensity

def contour_peri_int(contour, img):
    
    peri_list = []
    int_list = []
    i = 0
    while i<len(contour):
        con = contour[i]
        peri = len(con)
#         print(peri)
        peri_list.append(peri)
        
        mask = np.zeros(img.shape,np.uint8)
        cv2.drawContours(mask,[contour[i]],0,1,-1)
        pixelpoints = np.transpose(np.nonzero(mask))
        x, y = np.transpose(np.array(pixelpoints))
        intensity = img[x, y].mean()
        int_list.append(intensity)
        
        i = i+1
        
    return peri_list, int_list