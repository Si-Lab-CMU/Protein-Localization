import glob
import os 
import re
import shutil
import numpy as np

def folder_verify(folder):
    """Verify if the folder string is ended with '/' """
    if folder[-1]!="/":
        folder = folder+"/"
    return folder


def folder_file_num(folder, pattern = "*"):
    """
    How many files in the folder
    20240517 updated to use regex"""
    if folder[-1]!="/":
        folder = folder +"/"
    file_list =  np.array(sorted(glob.glob(folder+"*")))
    file_list = [f for f in file_list if re.search(pattern, f)]
    print("%s "%folder + "has %s files"%len(file_list))
    return(file_list)

def create_folder(folder):
    """Create a folder. If the folder exist, erase and re-create."""
    folder = folder_verify(folder)
    
    if os.path.exists(folder): # recreate folder every time. 
        print("%s folder exists.\n"%folder)
        pass
    else:
        os.makedirs(folder)
        print("%s folder is freshly created. \n"%folder)
    
    return folder

def create_folder_w_remove(folder):
    """Create a folder. If the folder exist, erase and re-create."""
    folder = folder_verify(folder)
    
    if os.path.exists(folder): # recreate folder every time. 
        shutil.rmtree(folder)
        os.makedirs(folder)
    else:
        os.makedirs(folder)
    print("%s folder is freshly created. \n"%folder)
    
    return folder