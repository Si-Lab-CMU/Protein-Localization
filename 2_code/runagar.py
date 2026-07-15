import util_local as ul
import os
import re
import sys
import subprocess

import nd2
import matplotlib.pyplot as plt
from tifffile import imsave

# Create a folder for all .nd2 files in the image folder

folder = "../1_data/"
nd2_fs = ul.folder_file_num(folder, r"\.nd2$")

fol_list = []
for f in nd2_fs:
    fol = f[:-4]
    ul.create_folder_w_remove(fol)
    fol_list.append(fol)
    
# Convert the nd2 file into tiffs in the curresponding folder
for i in range(len(nd2_fs)):
#     print(i)
    img = nd2.imread(nd2_fs[i])
    tif_fold = ul.folder_verify(fol_list[i])
    basename = os.path.basename(nd2_fs[i])[:-4]
    print(basename)
    print(img.shape)
    
    # For images with only phase contract images. 
    if len(img.shape)==3:
        for j in range(img.shape[0]):
            im = img[j]
            imsave(tif_fold+basename+"_XY%s_C1.tif"%(str(j).zfill(3)), im)
            
    elif len(img.shape)==4:
        for j in range(img.shape[0]):
            # j is looping through XY
            # k is looping chanels
            for k in range(img.shape[1]):
                im = img[j][k]
                imsave(tif_fold+basename+"_XY%s_C%s.tif"%(str(j).zfill(3), k+1), im)
                
    else:
        print("Image convertion dimentionn wrong. Please fix code.")
        sys.exit()
        
        
# Run Agar pad analysis for all folders
analysis_folder = "../3_analysis/"
debug_folder = "../4_debug/"
param_folder = "../5_param/"
analysis_folder = ul.create_folder(analysis_folder)
debug_folder = ul.create_folder(debug_folder)
param_folder = ul.create_folder(param_folder)

# Creating parameter files for each folder
param_list = []
for i in range(len(nd2_fs)):
    
    tif_fold = ul.folder_verify(fol_list[i])
    basename = os.path.basename(nd2_fs[i])[:-4]
    
    param = """
    # Parameter file for agarPad.py imaging script.

    ### Experimental information ###################################################
    # This will be used to identify the experiment when saving the output data
    # Path to .tif files. All the image files should be in the same folder

    experiment_name: "%s"
    path_to_tiffs: "%s"


    # Path to save analysis
    path_analysis: "%s"
    #Path to save any debug information (contour overlays). Must be full path
    path_debug: "%s"

    # Determine what analysis goes for what channels. The position of the items in corresponds to the channel number. You must have a phase contrast image, but it doesn't have to be the first in the list. The string is a keyword that should match the indicates the analysis to be peformed and should match image type:
    # pc_box : simple sizing based on box around contour
    # pc_thr : phase contrast sizing based on threshold
    # int : total and average fluorecence per cell
    # foci : foci position using maxmimum method
    # foci_lap : uses laplacian convolution method for foci detection
    channels: ['pc_thr', 'int'] #['pc_thr', 'foci', 'int'] #['foci_lap', 'pc_thr']
    # what is the file postfix for each analysis above, minus the .tif. You can do two analysis on one channe, ie, channels = ['pc_thr', 'int', 'foci'] and channel_postfixes = ['c1', 'c2', 'c2']. If you have no channel marker (ie, just phase images), use # # #  channel_postfixes = ['']. The string must immediately precede '.tif' in the file name.
    channel_postfixes: ['C2', 'C1'] #['C1', 'C3']

    # Quality control or debug mode boolean
    debug: False # toggle general debug mode
    debug_phase: False # toggle debug mode for phase analysis
    debug_foci: True # toggle foci detection debug
    debug_print_contour: True # toggle printing contour overlays

    # Image cropping. Indicate if the imaged should be cropped before analysis. The values indicate what part of the image should be analyzed. For example, row_start = 0, row_end = 1, col_start = 0, col_end 0.5, would be the top half of the image.
    # This is particular useful when there is a ROI that we want to analyze, or uneven light at the edges of the images. 
    crop_image: False
    row_start: 0 # value between 0 and 1, measured from image left edge
    row_end: 1 # value between 0 and 1, greater than row_start
    col_start: 0 # value between 0 and 1, measured from image top edge
    col_end: 0.5 # value between 0 and 1, greater than row_start

    # pixel to micron conversion
    # micron/pixel
    pxl2um: 0.065 # This is for 100x objective using the Andor Neo camera.

    ### Contour filtering parameters ###############################################
    # These parameters remove contours (suspected cells) based on size and shape, which come into play during intial contour finding (not during phase contrast sizing or fluorescent analysis).

    # Boundary removal. If cells are with this number of pixels from the edge of the image (after cropping), they will be removed. This excludes cells that may be cut off by the image edge. It is important when cropping images for further analysis
    boundary: 30
    # If the contour has an area less than this value in square pixels, discard.
    min_area: 300
    # If the ratio of the area of a bounding box around the countour to the area of the actual countour is greater than this value, discard. This excludes cells that do not lie straight or have a funny shape.
    rect_to_area: 1.5
    # Aspect ratio limits (length/width)
    min_aspect_ratio: 1 #20190712
    max_aspect_ratio: 1500
    # Countour width and length limits, all are in pixels
    con_min_w: 5 # min width
    con_max_w: 23 # max width
    con_min_l: 30  # min length
    con_max_l: 200 # max length

    ### pc_thr #####################################################################
    # These parameters are used during phase contrast sizing.

    # threshold from 0 to 1 to decide where cell boundary is during phase contrast sizing. We have been using 0.75, so don't change this.
    pc_int_thr: 0.75

    # length based measures
    lth_cut_up: 1500 # upper bound of acceptable cell length in microns
    lth_cut_low: 0 # lower bound of acceptable cell length in microns


    # width based measures

    #T01
    wth_cut_up: 1500 # upper bound of acceptable cell width in microns
    wth_cut_low: 0 # lower bound of acceptable cell width in microns

    ##T02
    #wth_cut_up: 0.696318888400507 # upper bound of acceptable cell width in microns
    #wth_cut_low: 0.673367793162412 # lower bound of acceptable cell width in microns

    ##T03
    #wth_cut_up: 0.800883613913336 # upper bound of acceptable cell width in microns
    #wth_cut_low: 0.547541342935809 # lower bound of acceptable cell width in microns

    ##T04
    #wth_cut_up: 0.815247636446283 # upper bound of acceptable cell width in microns
    #wth_cut_low: 0.538682919760021 # lower bound of acceptable cell width in microns
    #
    ##T05
    #wth_cut_up: 0.785095177501118 # upper bound of acceptable cell width in microns
    #wth_cut_low: 0.516092596205740 # lower bound of acceptable cell width in microns

    ##T06
    #wth_cut_up: 0.801135140521627 # upper bound of acceptable cell width in microns
    #wth_cut_low: 0.486233104698415 # lower bound of acceptable cell width in microns

    ##T07
    #wth_cut_up: 0.998249963628305 # upper bound of acceptable cell width in microns
    #wth_cut_low: 0.677007114012061 # lower bound of acceptable cell width in microns

    ##T08
    #wth_cut_up: 0.988117716530888 # upper bound of acceptable cell width in microns
    #wth_cut_low: 0.685255051842170 # lower bound of acceptable cell width in microns

    ##T09
    #wth_cut_up: 1.03544189537975 # upper bound of acceptable cell width in microns
    #wth_cut_low: 0.694769581726048 # lower bound of acceptable cell width in microns

    ##T10
    #wth_cut_up: 1.02266122518551 # upper bound of acceptable cell width in microns
    #wth_cut_low: 0.721413099033455 # lower bound of acceptable cell width in microns

    ##T11
    #wth_cut_up: 1.07045739424996 # upper bound of acceptable cell width in microns
    #wth_cut_low: 0.709333906584959 # lower bound of acceptable cell width in microns

    ##T12
    #wth_cut_up: 1.07045739424996 # upper bound of acceptable cell width in microns
    #wth_cut_low: 0.709333906584959 # lower bound of acceptable cell width in microns


    ### foci detection paramteters #################################################
    # 2016_0213_XTL520_Mglu
    # 0.9/18 c3

    # CC2518
    # 1.5/18 c3
    # 0.8/12 c2

    # T1
    # 1.0/18 c3
    # 0.8/12 c2
    foci_threshold: [1.5] # minimum ratio that max values must have over minimum values in neightboorhood
    foci_neighborhood: [18] # local area in image in which to determine max and min intensities

    ### foci_dog detection parameters ##############################################
    foci_log_minsig: 0 # minimum sigma of laplacian to convolve in pixels. Scales with minimum foci width to detect as 2*sqrt(2)*minsig
    foci_log_maxsig: 10# maximum sigma of laplacian to convolve in pixels. Scales with maximum foci width to convolve as above

    ##T1
    foci_log_thresh: 0.0005 # default: 0.002; absolute threshold in which laplacian well reaches to record potential foci
    foci_log_peak_med_ratio: 1.5 # default: 1.2; foci peaks must be this many times greater than median cell intensity. Think signal to noise ratio
    #

    """%(basename,tif_fold,analysis_folder,debug_folder)
    p_file = param_folder + "param_%s.yaml"%basename
    param_list.append(p_file)
    p = open(p_file, "w")
    p.write(param)
    p.close()
    
# Run agar pad analysis with each parameter file. 
for i in range(len(nd2_fs)):
    subprocess.call(["python agarPad.py -f '%s'"%param_list[i]], shell=True)
