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
    subject_pair = np.array([0, 1]) # index based 0, 
    base_path = '/home/dotr/paper/hyperalignment_rfMRI'
    data_type = "task_fc"  # "task_fc" or "rs_fc"
    sessions = [21, 22]
    
    input_path = os.path.join(base_path, "data_input")
    output_path = os.path.join(base_path, "data_output", "training", data_type)
    os.makedirs(output_path, exist_ok=True)

    # hyperalignment params
    hyp_mode = "connectivity-w-ROIs" 
    brain_mask = np.array(datasets.load_mni152_brain_mask(resolution=5).get_fdata()) 
    atlas = np.load(os.path.join(input_path,"harvard_oxford_atlas_5mm.npy"))
    reference = 0 # first subject
    stride = 9 
    radius = 5 # cube of 11x11x11(center voxel + 5 in all direction) (voxel unit)            
    svd_mode = "truncated" # "truncated" or "full" but "full" normally lead to instability and not convergence
    svd_variance_threshold = 0.8
    svd_max_components = 10 # 69 targets in the atlas 

    ### PREPARATION ###  
    # Prepare runs list differently based on data_type
    if data_type == "rs_fc":
        runs = [1, 14] 
    else:
        runs = None  # placeholder

    ### LOAD DATA ###
    data = []

    for subject in subject_pair:
        runs_data = []
        for session in sessions:
            if data_type == "rs_fc":
                runs_to_load = runs
            else:
                # Search input_path for files matching subj/session but runs NOT 1 or 14
                # File pattern: subjXX_timeseries_sessionYY_runZZ_5mm.nii.gz
                files = os.listdir(input_path)
                # regex to extract run number
                pattern = re.compile(rf"subj{subject+1:02d}_timeseries_session{session:02d}_run(\d+)_5mm.nii.gz")
                runs_to_load = []
                for f in files:
                    m = pattern.match(f)
                    if m:
                        run_num = int(m.group(1))
                        if run_num not in [1, 14]:
                            runs_to_load.append(run_num)

                runs_to_load = sorted(runs_to_load)

            # Load data for these runs and this session
            for run in runs_to_load:
                file_path = os.path.join(
                    input_path,
                    f"subj{subject+1:02d}_timeseries_session{session:02d}_run{run:02d}_5mm.nii.gz"
                )
                runs_data.append(nib.load(file_path).get_fdata())

        subject_data = np.concatenate(runs_data, axis=-1)  # concatenate along time dimension
        data.append(subject_data)
    
    data = np.stack(data)
    print("data shape: ", data.shape)
    
    ### PROCESS DATA FOR ALIGNMENT ###
    aligner = SL_HyperAlign(data_batch = data, 
                            subject_list = subject_pair,
                            reference=reference, 
                            hyp_mode=hyp_mode,
                            output_dir=output_path,
                            brain_mask=brain_mask, 
                            atlas=atlas)

    aligner.sl_align(stride=stride,
                 radius=radius,
                 svd_mode=svd_mode,
                 svd_max_components=svd_max_components,
                 svd_variance_threshold=svd_variance_threshold)


if __name__ == "__main__":
    main()