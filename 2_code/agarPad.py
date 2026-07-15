"""
Created on Tue Jan 13 11:06:17 2015
Edited on Dec 9-15 by jt -- added foci detection and independent ch analysis
Edited on Fri Sep 29 by jt -- added pc size analysis
Edited on Fri Sep 18 by jt
Edited on Thr Oct 15 by jt -- added parameter file support
Edited on 20151111 by jt -- functionlized everything and outsourced to
    other files. removed unused files
@author: yonggun
"""

# import modules
import os, sys, getopt, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import exposure
import cv2 # openCV
from termcolor import colored
from os import listdir
from os.path import isfile, join
import yaml # for parameter loading
import pprint # pretty printing
import re # regular expression
import pickle

# agarPad modules
from APLibrary import find_contours, filter_contours, index_cells, contour_area, plot_contours
from APPhase import pc_profile_sizing, contour_peri_int
from APLineProfile import line_profiling
from APFluorescence import fluor_intensity
from APFoci import foci_thr, foci_lap

# minor updated 
# Huijing 
import util_local as ul
import scipy.io as sio
################################################################################
### functions
# all the functions are in the supporting library files

################################################################################
### main script
if __name__ == "__main__":
    """agarPad
    agarPad calculates cellular size, fluorescence, and foci information from single cells imaged on agar pads or slides.
    A brightfield/pc image is always required to find initial cells, but fluorescence and foci imaging is optional.
    This script is called with reference to a parameters file which holds both the path to the image files, indicators for what analysis to do, as well as other parameters and switches.

    > python /path/to/agarPad.py -f '/path/to/parameters.yaml'

    The data is output as csv formatted .txt files, one for each type of analysis.

    See README and the .yaml file for more information.

    Options
    _______
    -f 'path/to/parameters.yaml'
        f for file. Put the path to the parameters file for this analysis
    -t
        t for test. Sets the parameters file to agarPad_params_test.yaml and
        outputs analysis on the test data set found in /test/TIFF. The
        output files can be checked against the standards in the test folder.

    Parameters
    ----------
    parameters.yaml : path to .yaml file
        Holds all parameter and path information. Description of the parameters are in that file.

    Returns
    -------
    See README
    """

    ############################################################################
    ### get arguments and parameters
    # switches
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:t")
    except getopt.GetoptError:
        print('No arguments detected.')

    for opt, arg in opts:
        if opt == '-f': # look for parameter file
            param_file = arg
        if opt == '-t': # run in test mode
            pass
    # param_file = './agarPad_params.yaml'

    # load the parameter file
    f = open(param_file)
    global params
    params = yaml.safe_load(f)
    f.close()
    if params['debug']:
        pprint.pprint(params)

    # sys.exit("Exit here.")
    
    # determine what analysis is to be done on what channels
    meas_pc_thr = False # profile threshold sizing
    meas_int = False # total and average fluorescence intensity
    meas_foci = False # foci detection
    for c, channel in enumerate(params['channels']):
        if channel == 'pc_box' or channel == 'pc_thr':
            channel_pc = params['channel_postfixes'][c]
            meas_pc_thr = True
            size_method = channel
        if channel == 'int':
            channel_int = params['channel_postfixes'][c]
            meas_int = True
            int_method = channel
        if channel == 'foci' or channel == 'foci_lap':
            channel_foci = params['channel_postfixes'][c]
            meas_foci = True
            foci_method = channel
            
    
    
    # Load the images
    path = params['path_to_tiffs'] # path to image folder

    # phase contrast images
    try:
        try: # try to see if there is a channel marker
            files_str = channel_pc + r'\.tif$'
            files_pc = [m for m in sorted(os.listdir(path)) if re.search(files_str, m, flags=re.I)] # re.I is for ignore case
            # print(files_pc)
        except: # if not then go for anything with a xy
            files_str = r'xy\d+?\.tif$'
            files_pc = [m for m in sorted(os.listdir(path)) if re.search(files_str, m, flags=re.I)] # re.I is for ignore case
    except:
        sys.exit("No phase contrast images found for channel " + channel_pc)

    # fluorecenct images for intensity
    if meas_int:
        try:
            files_str = channel_int + r'\.tif$'
            files_int = [m for m in sorted(os.listdir(path)) if re.search(files_str, m, flags=re.I)] # re.I is for ignore case
        except:
            sys.exit("No fluorescent images found for channel " + channel_int)

    # fluorecenct images for foci analysis
    if meas_foci:
        try:
            files_str = channel_foci + r'\.tif$'
            files_foci = [m for m in sorted(os.listdir(path)) if re.search(files_str, m, flags=re.I)] # re.I is for ignore case
        except:
            sys.exit("No fluor. foci images found for channel " + channel_foci)

    # print to terminal how many files were found
    print(colored("========================================================", 'red'))
    print("     ", colored(params['experiment_name'], 'green'))
    print(('Found %d files:' % len(files_pc)))
    print(colored("========================================================\n", 'red'))


    ############################################################################
    ### loop through all pictures, based on phase images

    # these array hold all data for all pictures, a is for all
    a_ind = [] # index
    a_lth = [] # length
    a_wth = [] # width
    a_area = [] # contour area
    line_profile =[] # line profile
    a_peri = [] # contour perimeter
    a_bfv = [] # contour bright field value, averge 

    if meas_int:
        avg_intensity = []
        tot_intensity = []
        background = []

    if meas_foci:
        a_indicies, a_disp_l, a_disp_w, a_foci_l, a_foci_w, a_foci_h1, a_foci_h2, a_foci_h3, a_foci_h4 = [], [], [], [], [], [], [], [], []

    for i, pc_name in enumerate(files_pc):
        # read phase image
        print(pc_name)
        img = plt.imread(path + pc_name) # read phase image

        if meas_int:
            fluoname = files_int[i] # get the fluorecent image name
            img_fluo = plt.imread(path + fluoname) # read fluorescent image

        if meas_foci:
            fociname = files_foci[i]
            img_foci = plt.imread(path + fociname)

        # crop image, both images should be the same size
        if params['crop_image']:
            rr, cc = img.shape
            
            img = img[int(params['row_start']*rr) : int(params['row_end']*rr),
                      int(params['col_start']*cc) : int(params['col_end']*cc)]

            if meas_int:
                img_fluo = img_fluo[int(params['row_start']*rr):int(params['row_end']*rr),
                                    int(params['col_start']*cc):int(params['col_end']*cc)]

            if meas_foci:
                img_foci = img_foci[int(params['row_start']*rr):int(params['row_end']*rr),
                                    int(params['col_start']*cc):int(params['col_end']*cc)]

        # equalize
        img = exposure.equalize_adapthist(img, clip_limit=0.005)

        ### run analysis #######################################################
        # find all the cells using contouring
        contours = find_contours(img, params)

        # filter contours for location, strange shapes, and sizes
        contours, lth_pc, wth_pc = filter_contours(img, contours, params, pc_name)

        ### phase contrast size analysis
        if meas_pc_thr:
            # contours, lth_pc, wth_pc = pc_profile_sizing(img, contours, params)
            contours, lth_pc, wth_pc, line_profile = line_profiling(img, img_fluo, contours, line_profile, params) #FS20210625

        # print number of cells and abort if there are none
        print( '===> Number of cells found is', len(lth_pc), '<===')
        if len(lth_pc) == 0:
            continue

        # determine contour areas
        con_areas = contour_area(contours)
        
        # determine contour perimeter and average intensity
        peri_list, bfv_list = contour_peri_int(contours, img)

        # give cells unique indexes based on their contours. This is critical for later analysis bookkeeping
        ind_pc = index_cells(contours, len(a_ind))

        # append indexes and size data and convert to microns
        a_ind = np.append(a_ind, ind_pc)
        a_lth = np.append(a_lth, lth_pc * params['pxl2um'])
        a_wth = np.append(a_wth, wth_pc * params['pxl2um'])
        a_area = np.append(a_area, con_areas * params['pxl2um']**2)
        
        a_peri = np.append(a_peri, peri_list) # contour perimeter
        a_bfv = np.append(a_bfv, bfv_list) # contour bright field value, averge

        ### fluorescence intensity analysis
        if meas_int:
            contours, avg_int_cells, tot_int_cells, mean_bg = fluor_intensity(img, img_fluo, contours, params)

            # append fluo data
            avg_intensity = np.append(avg_intensity, avg_int_cells)
            tot_intensity = np.append(tot_intensity, tot_int_cells)
            background = np.append(background, mean_bg)

        ### foci detection
        if meas_foci:
            if foci_method == 'foci':
                indicies, disp_l, disp_w, foci_l, foci_w, foci_h1, foci_h2, foci_h3, foci_h4 = foci_thr(img, img_foci, contours, ind_pc, params)
            elif foci_method == 'foci_lap':
                indicies, disp_l, disp_w, foci_l, foci_w, foci_h1, foci_h2, foci_h3, foci_h4 = foci_lap(img, img_foci, contours, ind_pc, params)

            # append data
            a_indicies = np.append(a_indicies, indicies)
            a_disp_l = np.append(a_disp_l, disp_l)
            a_disp_w = np.append(a_disp_w, disp_w)
            a_foci_l = np.append(a_foci_l, foci_l)
            a_foci_w = np.append(a_foci_w, foci_w)
            a_foci_h1 = np.append(a_foci_h1, foci_h1)
            a_foci_h2 = np.append(a_foci_h2, foci_h2)
            a_foci_h3 = np.append(a_foci_h3, foci_h3)
            a_foci_h4 = np.append(a_foci_h4, foci_h4)

        # display contours for whole picture
        contour_overlay = plot_contours(img, contours)

        # print contour image for quality control
        if params['debug_print_contour']:
            
            ul.create_folder(params["path_debug"])
            
            plt.figure(figsize=(16,16), dpi=80)
            plt.title('Contour Overlay of %s' % pc_name)
            plt.imshow(contour_overlay, cmap=plt.cm.gray, interpolation='nearest')
            # ---------------------------------------
            im_txt = ""
            xx = 0
            while xx<len(pc_name.split('.'))-1:
                im_txt = im_txt+pc_name.split('.')[xx]
                xx = xx+1
            img_title = params['path_debug'] + im_txt
            # --------------------------------------- In case xx has many "."
            # img_title = params['path_debug'] + pc_name.split('.')[0]
            cv2.imwrite(img_title+'_Overlay.jpg', contour_overlay)
            plt.close()

    # end loop through images
    ############################################################################
    ### collect information and print to files

    # calculate volume of cylinder with hemispherical ends
    a_vol = (np.array(a_lth)-np.array(a_wth))*np.pi*(np.array(a_wth)/2)**2+(4/3)*np.pi*(np.array(a_wth)/2)**3

    # calculate average values for printing
    l = np.mean(a_lth)
    w = np.mean(a_wth)
    v = np.mean(a_vol)

    # print information to terminal
    text = 'length = ' + str(l)[:6] + 'um, width = ' + str(w)[:6] + 'um, volume = ' + str(v)[:6] + 'um^3'
    print(colored("========================================================", 'red'))
    print( "     ", colored(params['experiment_name'], 'green'))
    print(colored("========================================================", 'red'))
    print(colored(text,'yellow'))
    print( colored("========================================================", 'red'))
    print('Writing files')

    ### write data
    # size data
    array = np.zeros([len(a_lth),7])
    array[:,0] = a_ind
    array[:,1] = a_lth
    array[:,2] = a_wth
    array[:,3] = a_vol
    array[:,4] = a_area
    
    array[:,5] = a_peri
    array[:,6] = a_bfv
    
    # freshly create the analysis folder every time
    ul.create_folder(params['path_analysis'])
            
    if meas_pc_thr:
        fname = params['path_analysis'] + params['experiment_name'] + '_' +  channel_pc + '_' + size_method + '.txt'
        np.savetxt(fname, array, fmt='%d,%f,%f,%f,%f,%f,%f')
    
    

    # fluorescence intensity data
    if meas_int:
        # shape data into array
        array = np.zeros([len(avg_intensity),5])
        array[:,0] = a_ind
        array[:,1] = avg_intensity
        array[:,2] = tot_intensity
        array[:,3] = np.mean(background)
        array[:,4] = tot_intensity/a_vol # updating an average over volumn in um^2

        # save file as csv
        fname = params['path_analysis'] + params['experiment_name'] + '_' + channel_int + '_' + int_method + '.txt'
        np.savetxt(fname, array, fmt='%d,%f,%f,%f,%f')

                #convert the line profile list to matlab cell array       
        fname = params['path_analysis'] + params['experiment_name'] + '_' +  channel_int + '_' + size_method + '_line_profile.mat'
        # np.savetxt(fname[:-4]+".txt", line_profile, fmt='%f,%f')
        # print(line_profile)
        # sio.savemat(fname, mdict={'line_profile': line_profile})

        # Save the array to a .pkl file
        filename = params['path_analysis'] + params['experiment_name'] + '_' +  channel_int + '_' + size_method + '_line_profile'+'.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(line_profile, file)

    if meas_foci:
        
        # shape data into array
        array = np.zeros([len(a_indicies),9])
        array[:,0] = a_indicies
        array[:,1] = a_disp_l
        array[:,2] = a_disp_w
        array[:,3] = a_foci_l
        array[:,4] = a_foci_w
        array[:,5] = a_foci_h1
        array[:,6] = a_foci_h2
        array[:,7] = a_foci_h3
        array[:,8] = a_foci_h4

        # save file as csv
        fname = params['path_analysis'] + params['experiment_name'] + '_' + channel_foci + '_' + foci_method + '.txt'
        np.savetxt(fname, array, fmt='%d,%f,%f,%f,%f,%f,%f,%f,%f')

    print('Finished')
