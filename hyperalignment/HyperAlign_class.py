from HyperAlign_utils import *
from math import prod
import os 
import cupy as cp
import numpy as np
import h5py

class HyperAlign:
    def __init__(self, 
                 data_batch,
                 subject_list, 
                 reference, # "common" or one chosen subject 
                 hyp_mode, # "spatial", "connectivity", "connectivity-w-ROIs"
                 output_dir,
                 atlas=None,
                 brain_mask=None):
        """
        Hyperalignment for multi-subject fMRI data.

        Aligns subject data to a common representational  or to one reference subject using a specified
        hyperalignment mode ("spatial", "connectivity", "connectivity-w-ROI" - need atlas). 
        Supports brain masking, preprocessing data for hyperalignment, and output saving.
        """

        self.subject_list = subject_list        
        self.reference = reference
        self.hyp_mode = hyp_mode
        self.brain_mask = brain_mask
        self.atlas = atlas 
        
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.preprocessed_dir = os.path.join(self.output_dir, "preprocessed")
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        
        if len(data_batch.shape) == 5: # shape subjects x image shape (3 dim) x time/ROI/stimuli
            print("Assuming input data has shape n_subjects x spatial xyz x dim5. Continue with preprocessing.")
            self.data_batch = data_batch

            # check if preprocessed data already exist 
            missing_subjects = []
            for subject in subject_list:
                subj_str = f'{subject+1:02d}'
                file_to_check = os.path.join(self.preprocessed_dir, f"subj{subj_str}_{self.hyp_mode}.h5")
            
                if not os.path.exists(file_to_check):
                    missing_subjects.append(subject)
                    
            self.missing_subjects = np.asarray(missing_subjects)
            # preprocess missing subjects
            if len(missing_subjects) > 0:
                print("Preprocessing missing data")
                indices = np.where(np.isin(self.subject_list, missing_subjects))[0]
                selected = data_batch[indices]
                self.preprocess(data=selected)

            print("Preprocessing is done")
            
        elif len(data_batch.shape) == 3: # assume data is preprocessed 
            print("Assuming input data was preprocessed and has shape n_subjects x dim1 x dim2")
            print("No preprocessing was done")

    def preprocess(self, data):
        """
        Preprocess fMRI data: reshape, mask, and normalize + 
        + (if hyperalignment done with connectivity matrices) compute connectivity, normalize
        """
        
        # Step 1: Reshape data
        data_reshaped = reshape_data(self.data_batch)
        print(data_reshaped.shape)

        # Step 2: Apply brain mask
        if self.brain_mask is not None:
            data_masked = brain_masking(data_reshaped, self.brain_mask)

        # Step 3: Normalize data
        data_normalized = normalize(data_masked, method="zscore", within="columns")

        # Step 4: Compute connectivity matrices
        if self.hyp_mode == "spatial":
            preprocessed_data = data_normalized
            
        elif self.hyp_mode == "connectivity-w-ROIs":
            targets = define_targets(data_normalized, self.atlas, self.brain_mask)
            # calculate connectivity matrices
            con_matrices = calculate_connectivity_matrix(data_normalized, targets)
            # normalize
            preprocessed_data = normalize(con_matrices, method="fisher_z", within="columns")

        elif self.hyp_mode == "connectivity":
            # calculate connectivity matrices
            con_matrices = calculate_connectivity_matrix(data_normalized, target_batch=None)
            # normalize
            preprocessed_data = normalize(con_matrices, method="fisher_z", within="columns")

        print("preprocessed data shape: ", preprocessed_data.shape)
        # SAVE
        for i, subj in enumerate(self.missing_subjects):
            subj_str = f'{subj.item()+1:02d}' 
            file_name = os.path.join(self.preprocessed_dir, f'subj{subj_str}_{self.hyp_mode}.h5') 
            with h5py.File(file_name, 'w') as f: f.create_dataset('prep_data', data=preprocessed_data[i, :, :])
    
    def align(self, cal_perf_metrics=True, 
              svd_mode="truncated", svd_max_components=100, svd_variance_threshold=0.8,
              preprocessed_data=None,):
        """
        Perform hyperalignment across subjects.

        Parameters
        ----------
        cal_perf_metrics (bool) If True, computes alignment quality metrics. Default is True.
        svd_mode (str) SVD mode for dimensionality reduction ("truncated" or "full").
        svd_components (int) Number of components to retain if using truncated SVD. Default is 30.
        svd_variance_threshold (int) Variance threshold for adaptive component selection. Default is 0.8.
        preprocessed_data (array) preprocessed input data to align. 
                                  If None, uses preprocessed data done by the Object itself (in self.preprocessed_dir)

        Returns
        -------
        dict: Dictionary containing:
        - 'aligned_data': aligned fMRI data
        - 'Rs': transformation matrices
        - 'common_model': shared model reference
        - 'perf_metrics': performance metrics (if calculated)
        """
        if preprocessed_data is None: 
            # search for preprocessed data in self.preprocessed_dir and load 
            all_subjects = [] 
            for subj in self.subject_list: 
                subj_str = f'{subj+1:02d}' 
                file_path = os.path.join(self.preprocessed_dir, f"subj{subj_str}_{self.hyp_mode}.h5") 
                with h5py.File(file_path, "r") as f: 
                    all_subjects.append(np.array(f["prep_data"])) 
                
            preprocessed_data = np.stack(all_subjects)

        n_subjects, _, n_dim3 = preprocessed_data.shape

        # PREPARE STORAGE
        Rs = cp.empty((n_subjects, n_dim3, n_dim3))
        aligned_data = cp.empty(preprocessed_data.shape)
        common_model = cp.empty(preprocessed_data.shape[1:])

        # HYPERALIGNMENT TO COMMON MODEL
        if self.reference == "common":
            # FIRST ITERATION: Iteratively align each subject to mean of previously aligned data
            aligned_data[0, :, :] = preprocessed_data[0, :, :]

            for i, data in enumerate(preprocessed_data[1:, :, :]):
                reference = cp.mean(aligned_data, axis=0)
                
                R = calculate_transmat(data, reference, svd_mode, 
                                       max_components=svd_max_components, 
                                       variance_threshold=svd_variance_threshold)
                aligned_data[i+1,:,:] = data @ R
            # SUBSEQUENT ITERATIONS: Refine alignment by aligning each subject to the mean of all previously aligned data
            for iter in range(2):
                reference = cp.mean(aligned_data, axis=0)

                for i, data in enumerate(preprocessed_data):
                    R = calculate_transmat(data, reference, svd_mode,
                                           max_components=svd_max_components, 
                                           variance_threshold=svd_variance_threshold)
                    aligned_data[i,:,:] = data @ R
                    
                    # Store transformation matrix if last iteration
                    if iter == 1:
                        Rs[i] = R 
                        
            # COMMON MODEL as the last iteration's reference
            common_model = reference
        
        # HYPERALIGNMENT TO A SPECIFIC SUBJECT
        else:
            reference = preprocessed_data[self.reference, :, :]
            for i, data in enumerate(preprocessed_data):
                R = calculate_transmat(data, reference, svd_mode,
                                       max_components=svd_max_components,
                                       variance_threshold=svd_variance_threshold)
                aligned_data[i,:,:] = data @ R
                Rs[i] = R

            common_model = reference
        
        # CALCULATE PERFORMANCE METRICS
        perf_metrics = {}
        if cal_perf_metrics:
            # disparity 
            perf_metrics["initial_disparity"] = calculate_disparity_matrix(preprocessed_data).get()  
            perf_metrics["final_disparity"] = calculate_disparity_matrix(aligned_data).get()

            # isc
            perf_metrics["initial_isc"] = calculate_isc(preprocessed_data, 
                                                        isc_type="row", 
                                                        ref_subject=self.reference, 
                                                        fisher_z=False).get()
            perf_metrics["final_isc"] = calculate_isc(aligned_data, 
                                                      isc_type="row", 
                                                      ref_subject=self.reference, 
                                                      fisher_z=False).get()

        return {'aligned_data': aligned_data.get(),
                'Rs': Rs.get(),
                'common_model': common_model.get(),
                'perf_metrics': perf_metrics}


class SL_HyperAlign(HyperAlign):   

    """
    Searchlight-based Hyperalignment.
    Extends HyperAlign to apply alignment within local spherical regions defined by a given radius and stride.
    """         
    
    def __init__(self, data_batch, subject_list, 
                 reference, hyp_mode, output_dir,
                 atlas=None, brain_mask=None, image_shape=None):
        
        """
        Initialize the SL_HyperAlign class.
        """
        super().__init__(data_batch, subject_list, reference, hyp_mode, output_dir, atlas, brain_mask)
        self.missing_subjects = getattr(self, "missing_subjects", None)

        if self.data_batch is not None and self.data_batch.ndim == 5:
            self.image_shape = self.data_batch.shape[1:-1]
        elif image_shape is not None:
            self.image_shape = image_shape
        else:
            raise ValueError("Please provide shape of the original image (x,y,z)")

        self.n_subject = len(self.subject_list)

    def sl_align(self, stride, radius, correct_R=True, cal_perf_metrics=True, 
                 svd_mode="truncated", svd_max_components=100, svd_variance_threshold=0.8,
                 preprocessed_dir=None):
        """
        Run searchlight-based hyperalignment across the brain.

        Applies local alignment within spherical searchlights, aggregates transformation
        matrices, and saves aligned data, transformation matrices, common model and local performance metrics.

        Parameters
        ----------
        correct_R (bool) Whether to zscore within columns for R.
        cal_perf_metrics (bool) Whether to compute and save local performance metrics. Default is True.
        svd_mode (str) SVD strategy for dimensionality reduction. Default is "truncated".
        svd_components (int) Number of SVD components to retain. Default is 30.
        svd_variance_threshold (float) Variance threshold for adaptive SVD. Default is 0.8.
        preprocessed_dir: directory of preprocessed data - if you already have preprocessed data
                          Default: None, preprocessed data taken from self.preprocessed_dir
        """

        # SET UP STORAGE
        # directory 
        ha_res_dir = os.path.join(self.output_dir, "ha_result")
        os.makedirs(ha_res_dir, exist_ok=True)
        
        # transformation matrices and open preprocessed data 
        prep_files = []
        R_files = []

        for subject in self.subject_list:
            subj_str = f"{subject+1:02d}"

            # --- Preprocessed input files ---
            if preprocessed_dir == None:
                prep_file = os.path.join(self.preprocessed_dir, f'subj{subj_str}_{self.hyp_mode}.h5')
            else:
                print("==========================================================")
                print("WARNING: You are using your own preprocessed data.")
                print("The searchlight hyperalignment function requires the following:")
                print("1. The data must be stored as an HDF5 (.h5) file (requires h5py).")
                print("2. The file must contain a dataset with the key 'prep_data'.")
                print("3. File naming format must be: 'subjXX_hypmode.h5',")
                print("   where XX is the subject number with leading zeros, and hypmode is your hyperalignment mode.")
                print("\nExample of a valid file name: 'subj01_spatial.h5'; 'subj02_connectivity.h5', 'subj03_connectivity-w-ROIs.h5'")
                print("==========================================================\n")
                
                prep_file = os.path.join(preprocessed_dir, f'subj{subj_str}_{self.hyp_mode}.h5')

            prep_files.append(h5py.File(prep_file, 'r'))
                
            # --- Transformation matrix output files ---
            ref_str = f"{self.subject_list[self.reference]+1:02d}" if self.reference != "common" else self.reference
            R_path = os.path.join(ha_res_dir, f'R_subj{subj_str}_{self.hyp_mode}_ref-{ref_str}.h5')
            R_file = h5py.File(R_path, 'w')

            if self.brain_mask is None:
                R_file.create_dataset('R', data=np.eye(prod(self.image_shape), dtype='float32'))
            else:
                R_file.create_dataset('R', data=np.eye(int(np.sum(self.brain_mask)), dtype='float32'))

            R_files.append(R_file)

        # common model
        if self.reference == "common":
            shape = prep_files[0]['prep_data'].shape
            common_model_file = h5py.File(os.path.join(ha_res_dir, f'common-model_{self.hyp_mode}.h5'), 'w')
            common_model_file.create_dataset('common-model', data=np.zeros(shape=shape, dtype='float32'))

        # performance metrics
        perf_metrics = {}

        # SEARCHLIGHT ALIGNMENT
        # Generate searchlight centers
        x, y, z = cp.meshgrid(cp.arange(self.image_shape[0]), 
                              cp.arange(self.image_shape[1]),
                              cp.arange(self.image_shape[2]), indexing="ij")
        centers = cp.column_stack((x.ravel(), y.ravel(), z.ravel()))[::stride]
        total_centers = len(centers)
        print("Number of centers: ", total_centers)

        # Hyperalign each searchlight
        last_printed = 0
        for center_idx, center in enumerate(centers):

            # Print progress
            progress = (center_idx) / total_centers * 100
            if progress - last_printed >= 5:
                last_printed += 5
                print(f"Processed {progress:.2f}% searchlights")

            # Get neighboring voxels and keep only brain coordinates
            local_image_coords = get_searchlight_indices(center, self.image_shape, radius)
            local_image_indices = [voxel_tracking(tuple(coord), self.image_shape, "coord_to_index")
                                    for coord in local_image_coords] # before masking
            index_mapping = create_index_mapping(self.brain_mask)  # Map before and after masking
            local_image_indices = map_indices(local_image_indices, index_mapping).get()  # Apply mapping

            # Remove -1 values - correspond to non-brain voxels, then skip loop if no indices found 
            local_image_indices = [idx for idx in local_image_indices if idx != -1]
            if not local_image_indices or len(local_image_indices) < 10:
                continue 

            # Extract local data
            if self.hyp_mode in ["spatial", "connectivity-w-ROIs"]:
                data_shape = prep_files[0]['prep_data'][:, local_image_indices].shape
                
            elif self.hyp_mode == "connectivity":
                i, j = np.ix_(local_image_indices, local_image_indices)
                data_shape = prep_files[0]['prep_data'][i, j].shape

            local_image_data = cp.empty((self.n_subject, *data_shape), dtype=cp.float32)

            for s in range(self.n_subject):
                if self.hyp_mode in ["spatial", "connectivity-w-ROIs"]:
                    preprocessed_data = prep_files[s]['prep_data'][:, local_image_indices]
                elif self.hyp_mode == "connectivity":
                    i, j = np.ix_(local_image_indices, local_image_indices)
                    preprocessed_data = prep_files[s]['prep_data'][i, j]
                
                local_image_data[s] = cp.asarray(preprocessed_data)

            # Perform local hyperalignment
            local_results = self.align(preprocessed_data=local_image_data, 
                                       cal_perf_metrics=cal_perf_metrics,
                                       svd_mode=svd_mode,
                                       svd_max_components=svd_max_components,
                                       svd_variance_threshold=svd_variance_threshold)
            
            del local_image_data

            # Update transformation matrices and common model
            for s in range(self.n_subject):
                current_R = R_files[s]['R'][:]

                for i, idx in enumerate(local_image_indices):
                    for j, jdx in enumerate(local_image_indices):
                        current_R[idx, jdx] += local_results['Rs'][s, i, j]
                        
                R_files[s]['R'][:] = current_R

            if self.reference == "common":
                for i, idx in enumerate(local_image_indices):
                    common_model_file['common-model'][:, idx] += local_results['common_model'][:, i]
            

            # Store local performance metrics
            if cal_perf_metrics: 
                mask = ~np.eye(local_results['perf_metrics']['initial_disparity'].shape[0], dtype=bool)
                
                for key in ['initial_local_disparity', 'final_local_disparity']:
                    base_key = key.replace('local_', '')
                    mean_value = local_results['perf_metrics'][base_key][mask].mean()
                    perf_metrics.setdefault(key, []).append(mean_value)
                
                for key in ['initial_local_isc', 'final_local_isc']:
                    base_key = key.replace('local_', '')
                    metric = local_results['perf_metrics'][base_key]
                    mean_value = np.nanmean(metric) if metric.size != 0 and np.any(~np.isnan(metric)) else np.nan
                    perf_metrics.setdefault(key, []).append(mean_value)

        print("Processed 100% searchlights")

        # CALCULATE AND SAVE FINAL RESULTS
        for i, subj in enumerate(self.subject_list):
            # normalize transformation matrices
            R = R_files[i]['R'][:]
            if correct_R:
                R = normalize(cp.array(R), within="columns").get()
            
            # save 
            R_files[i]['R'][:] = R

            # perform alignment and save aligned data
            preprocessed_data = prep_files[i]['prep_data'][:]
            aligned_data = preprocessed_data @ R
            
            subj_str = f"{subj+1:02d}" 
            ref_str = f"{self.subject_list[self.reference]+1:02d}" if self.reference != "common" else self.reference
            aligned_path = os.path.join(ha_res_dir, f'aligned_subj{subj_str}_{self.hyp_mode}_ref-{ref_str}.h5')
            with h5py.File(aligned_path, 'w') as f:
                f.create_dataset('aligned_data', data=aligned_data)

        # save performance metrics 
        if cal_perf_metrics:
            metrics_file_path = os.path.join(ha_res_dir, f'perf_metrics_{self.hyp_mode}.h5')
            with h5py.File(metrics_file_path, 'w') as f:
                for key, value in perf_metrics.items():
                    f.create_dataset(key, data=value)
                    
        # save common model
        if (self.reference == "common") & correct_R:
            common_model = common_model_file['common-model'][:]
            common_model = normalize(cp.array(common_model), within="columns").get()
            common_model_file['common-model'][:] = common_model

        # Close all files
        for f in R_files:
            f.close()

        for f in prep_files:
            f.close()

        if self.reference == "common":
            common_model_file.close()

        clean_gpu_memory(keep_vars=None)