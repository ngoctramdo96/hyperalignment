### LOAD PACKAGES ### 
import re
import os 
import numpy as np
from nilearn import datasets 
import nibabel as nib

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hyperalignment')))
from HyperAlign_class import * 

def main():

    ## USER's INPUTS
    # paths
    subject_pair = np.array([6,7]) # index based 0
    base_path = '/home/dotr/paper/hyperalignment_rfMRI'
    data_type = "beta_task"
    
    input_path = os.path.join(base_path, "data_input") # input data from other project
    output_path = os.path.join(base_path, "data_output", "test", data_type)
    os.makedirs(output_path, exist_ok=True)

    # hyperalignment params
    hyp_mode = "spatial"
    brain_mask = np.array(datasets.load_mni152_brain_mask(resolution=5).get_fdata()) 
    atlas = np.load(os.path.join(input_path,"harvard_oxford_atlas_5mm.npy"))
    reference = 0 # first subject

    ### LOAD DATA ###
    data = []
    for subject in subject_pair:
        file_path = os.path.join(input_path, f"subj{subject+1:02d}_task_test.nii.gz")
        subject_data = nib.load(file_path).get_fdata()
        print(subject_data.shape)
        data.append(subject_data)

    data = np.stack(data)
    print("Data shape: ", data.shape)
    
    ### PROCESS DATA FOR ALIGNMENT ###
    aligner = SL_HyperAlign(data_batch = data, 
                            subject_list = subject_pair,
                            reference=reference, 
                            hyp_mode=hyp_mode,
                            output_dir=output_path,
                            brain_mask=brain_mask, 
                            atlas=atlas)


if __name__ == "__main__":
    main()