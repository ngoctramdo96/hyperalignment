#%%
import cupy as cp
import gc
import h5py
import numpy as np

#%%
def reshape_data(data_batch):
    """
    Reshapes 5D fMRI data from (subject, x, y, z, t) to 3D (subject, t, voxel) format.

    Parameters:
        data_batch (array): Input fMRI data with shape (subject, x, y, z, t).

    Returns:
        cp.ndarray: Reshaped data with shape (subject, t, voxel).
    """
    batch_size, x, y, z, t = data_batch.shape
    
    # Collapse spatial dims into voxels
    data_reshaped = data_batch.reshape(batch_size, -1, t)
    
    # Transpose last two dims (timepoints, voxels)
    data_reshaped = data_reshaped.transpose(0, 2, 1)
    
    return data_reshaped

# ##### TEST #####
# import numpy as np
# data1_cpu_4D = np.random.rand(41, 48, 39, 300)
# data2_cpu_4D = np.random.rand(41, 48, 39, 300)
# data3_cpu_4D = np.random.rand(41, 48, 39, 300)


# data1_gpu_4D = cp.array(data1_cpu_4D)
# data2_gpu_4D = cp.array(data2_cpu_4D)
# data3_gpu_4D = cp.array(data3_cpu_4D)

# all_data_gpu_4D = cp.stack([data1_gpu_4D, data2_gpu_4D])

# reshaped_data_gpu = reshape_data(all_data_gpu_4D)

# from hyperalign_utils import reshape_data
# reshaped_data_cpu = reshape_data([data1_cpu_4D, data2_cpu_4D])

# for i in range(len(reshaped_data_cpu)):
#     assert np.allclose(reshaped_data_cpu[i], cp.asnumpy(reshaped_data_gpu[i])), "GPU and CPU reshaped data have different results"

# for i in range(len(reshaped_data_cpu)):
#     print(np.all(reshaped_data_cpu[i] == cp.asnumpy(reshaped_data_gpu[i])))
#     print(reshaped_data_cpu[i].shape)
#     print(reshaped_data_gpu[i].shape)

# print(reshaped_data_cpu[0])
# print(reshaped_data_gpu[0])

#%%
def calculate_matrix_correlation(data1, data2, mode):
    """
    Calculates the correlation between corresponding rows or columns of two matrices.

    Parameters:
        data1, data2 (cp.ndarray): Input matrices.
        mode (str): Use "column" to compute correlation between corresponding columns,
                    or "row" to compute between corresponding rows.

    Returns:
        cp.ndarray: 1D array of correlation coefficients between the corresponding
                    rows or columns of `data1` and `data2`.
    """
    if mode == "column":
        axis = 0
    elif mode == "row":
        axis = 1
    else:
        raise ValueError("mode must be column or row")
    
    # Demean
    data1_mean = cp.mean(data1, axis=axis, keepdims=True)
    data2_mean = cp.mean(data2, axis=axis, keepdims=True)
    data1_demeaned = data1 - data1_mean
    data2_demeaned = data2 - data2_mean

    # Calculate correlation
    numerator = cp.sum(data1_demeaned * data2_demeaned, axis=axis)
    denominator = cp.sqrt(cp.sum(data1_demeaned ** 2, axis=axis) * cp.sum(data2_demeaned ** 2, axis=axis))

    correlations = numerator / denominator
    # correlations = cp.nan_to_num(correlations, nan=0.0)

    return correlations

# ##### TEST #####
# corr_col_gpu = calculate_matrix_correlation(reshaped_data_gpu[0], reshaped_data_gpu[1], mode="column")
# print("Column correlation:", corr_col_gpu)
# corr_row_gpu = calculate_matrix_correlation(reshaped_data_gpu[0], reshaped_data_gpu[1], mode="row")
# print("Row correlation:", corr_row_gpu)

# from hyperalign_utils import calculate_matrix_correlation

# corr_col_cpu = calculate_matrix_correlation(reshaped_data_cpu[0], reshaped_data_cpu[1], mode="column")
# print("Column correlation:", corr_col_cpu)
# corr_row_cpu = calculate_matrix_correlation(reshaped_data_cpu[0], reshaped_data_cpu[1], mode="row")
# print("Row correlation:", corr_row_cpu)

# assert np.allclose(corr_row_cpu, cp.asnumpy(corr_row_gpu)), "GPU and CPU reshaped data have different results"
# assert np.allclose(corr_col_cpu, cp.asnumpy(corr_col_gpu)), "GPU and CPU reshaped data have different results"

# import matplotlib.pyplot as plt
# diff = corr_col_cpu-cp.asnumpy(corr_col_gpu)
# print(np.max(diff))
# plt.plot(diff[:1000])

#%%
def brain_masking(data_batch, brain_mask):
    """
    Applies a brain mask to fMRI data to exclude non-brain voxels.

    Parameters:
            data_batch (array): A 3D array of shape (subject, time, voxel)
            brain_mask (array): A 1D or 3D array indicating brain voxels.

    Returns
        array: Masked fMRI data of shape (subject, time, n_voxels)
                    `n_voxels` is the number of voxels within the brain mask.
    """

    # Convert brain_mask to 1D boolean array
    brain_mask = np.asarray(brain_mask)
    if brain_mask.ndim == 3:
        brain_mask = brain_mask.flatten().astype(bool)
    elif brain_mask.ndim != 1:
        raise ValueError("brain_mask must be a 3D or 1D array")
    
    # Mask
    masked_data = data_batch[:, :, brain_mask]

    return masked_data


# ##### TEST #####
# from nilearn import datasets
# resolution = 5 
# mni_brain_mask = cp.array(datasets.load_mni152_brain_mask(resolution=5).get_fdata())
# masked_gpu_data = brain_masking(reshaped_data_gpu, brain_mask=mni_brain_mask)

# from hyperalign_utils import brain_masking
# masked_cpu_data = brain_masking(reshaped_data_cpu, brain_mask=cp.asnumpy(mni_brain_mask))

# assert np.allclose(masked_cpu_data, cp.asnumpy(masked_gpu_data)), "GPU and CPU have different results"

# for i in range(len(masked_cpu_data)):
#     assert np.all(masked_gpu_data[i].get() == masked_cpu_data[i]) , "GPU and CPU have different results"
#     diff = masked_gpu_data[i].get() - masked_cpu_data[i]
#     print(np.unique(diff))

#     print(masked_gpu_data[i].shape)
#     print(masked_cpu_data[i].shape)

#%%
def normalize(data_batch, method='zscore', within='columns'):
    """
    Normalizes each 2D data matrix in a 3D batch using the specified method.

    Parameters
        data_batch (array): A 3D array of shape (subject, dim1, dim2)
        method (str): Normalization method: 'zscore' (default), 'euclidean', 'fisher_z'
        within (str): Direction along which to normalize:
            - 'columns' (default): Normalize across rows (i.e., within each column) [axis=1].
            - 'rows': Normalize across columns (i.e., within each row) [axis=2].
    Returns
        array: Normalized data with the same shape as `data_batch`.

    """
    valid_methods = {"euclidean", "zscore", "fisher_z"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")

    if within == 'columns':
        axis = 1
    elif within == 'rows':
        axis = 2
    else:
        raise ValueError("along must be 'rows' or 'columns'")

    # Fisher 
    if method == "fisher_z":
        if not np.all((data_batch >= -1) & (data_batch <= 1)):
            raise ValueError("fisher_z method requires data values strictly between -1 and +1")
        
        data_batch[data_batch == 1] = 0.99
        data_batch[data_batch == -1] = -0.99
        normalized_data_batch = np.arctanh(data_batch)
        return normalized_data_batch

    # Center the data along the specified axis
    data_centered = data_batch - np.mean(data_batch, axis=axis, keepdims=True)

    # Compute normalization factor
    if method == 'euclidean':
        data_norm = np.linalg.norm(data_centered, axis=axis, keepdims=True)
        data_norm = np.where(data_norm == 0, 1e-10, data_norm)  # avoid division by zero
        normalized_data_batch = data_centered / data_norm

    elif method == "zscore":
        data_std = np.std(data_centered, axis=axis, keepdims=True)
        data_std = np.where(data_std == 0, 1e-10, data_std)
        normalized_data_batch = data_centered / data_std

    return normalized_data_batch

##### TEST ####

# data_gpu_small = cp.array([[[500,3,4,1],
#                       [2,3,1,7]],
#                       [[2,3,4,1],
#                       [2,3,1,7]]])

# data_gpu_small = cp.array([[[-0.5, 0.3, 0.4, 0.99],
#                       [ 0.2, -0.3, 0.99, 0.7]],
#                       [[0.2, 0.3, 0.4, 0.99],
#                       [ 0.2, 0.3, 0.1, 0.7]]])


# data_cpu = data_gpu_small.get()

# test_mode = "fisher_z"
# within = "columns"
# normalized_data_gpu_small = normalize(data_gpu_small, method=test_mode, within=within)

# normalized_data_gpu = normalize(reshaped_data_gpu, method="zscore", within=within)

# from hyperalign_utils import normalize
# if within == "columns":
#     axis = 0
# else:
#     axis = 1

# normalized_data_cpu_small = normalize(data_cpu, method=test_mode, axis=axis)
# normalized_data_cpu = normalize(reshaped_data_cpu, method="zscore", axis=axis)

# normalized_data_cpu_small = np.stack(normalized_data_cpu_small)

# assert np.allclose(normalized_data_cpu_small, cp.asnumpy(normalized_data_gpu_small)), "GPU and CPU have different results"

# for i in range(len(normalized_data_cpu_small)):
#     diff = normalized_data_gpu_small[i].get() - normalized_data_cpu_small[i]
#     plt.plot(diff.flatten())
#     plt.show()
#     print(diff.max())
#     print(diff.min())


#%%
def calculate_connectivity_matrix(data_batch, target_batch=None):
    """
    Calculates correlation-based connectivity matrices from fMRI time-series data.

    Parameters:
        data_batch (array): A 3D array of shape (subjects, timepoints, voxels).
        target_batch (array): A 3D array of shape (subjects, timepoints, num_targets).
            If provided, the function computes the correlation between each target and each voxel in `data_batch`. 
            If None (default), voxel-to-voxel correlation is computed.

    Returns:
        conn_mats (array): A 3D CuPy array of connectivity matrices for each subject.
        Shape:
        - (subject, voxels, voxels) if `target_batch` is None.
        - (subject, num_targets, voxels) if `target_batch` is provided.

    Notes:
        - NaNs in the resulting correlation matrices are replaced with 0.0.
        - Diagonal values are set to 1.0 when computing voxel-to-voxel connectivity.
    """
    num_subj, _ , num_voxels = data_batch.shape

    # voxel-to-voxel connectivity
    if target_batch is None:
        conn_mats = np.empty((num_subj, num_voxels, num_voxels), dtype=cp.float32)

        for i in range(num_subj):
            data = data_batch[i]
            corr = np.corrcoef(data, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)
            conn_mats[i] = corr

    # target-to-voxel connectivity
    else:
        num_targets = target_batch.shape[2]
        conn_mats = np.empty((num_subj, num_targets, num_voxels), dtype=cp.float32)

        for i in range(num_subj):
            data = data_batch[i]
            targets = target_batch[i]
            corr = np.corrcoef(targets, data, rowvar=False)[:num_targets, num_targets:]
            corr = np.nan_to_num(corr, nan=0.0)
            conn_mats[i] = corr

    return conn_mats

##### TEST #####
# target1 = cp.random.rand(300, 69)
# target2 = cp.random.rand(300, 69)

# target1 = cp.random.rand(2, 4)
# target2 = cp.random.rand(2, 4)

# target_batch = None
# target_batch = cp.stack([target1, target2])

# connmat_gpu = calculate_connectivity_matrix(normalized_data_gpu_small, target_batch)

# from hyperalign_utils import calculate_connectivity_matrix
# target_list = None
# target_list = [cp.asnumpy(target1), cp.asnumpy(target2)]
# connmat_cpu = calculate_connectivity_matrix(normalized_data_cpu_small, target_list=target_list)

# for i in range(len(connmat_cpu)):
#     assert np.allclose(connmat_cpu[i], cp.asnumpy(connmat_gpu[i])), "GPU and CPU have different results"
#     diff = cp.asnumpy(connmat_gpu[i]) - connmat_cpu[i]
#     plt.plot(diff.flatten())
#     plt.show()
#     print(diff.max())
#     print(diff.min())


#%%
import warnings
warnings.filterwarnings('ignore', module='tensorly.*')

import tensorly as tl
tl.set_backend('cupy')

def calculate_transmat(data, reference, svd_mode, variance_threshold=0.8, max_components=100):
    
    """
    Computes an orthogonal transformation matrix using Procrustes alignment
    between `data` and `reference`, optionally using variance-based SVD truncation.

    Parameters:
        data (cp.ndarray): 2D array. Matrix to be aligned.
        reference (cp.ndarray): 2D array. Target matrix.
        svd_mode (str): Method for SVD decomposition. Options:
            - 'full': Full SVD using CuPy.
            - 'truncated': Truncated SVD using `tl.tenalg.svd_interface`.
        n_components : int, optional
            Number of SVD components to retain if `variance_threshold` is not set.
            Default is 20.
        variance_threshold : float, optional
            If set (0 < threshold ≤ 1), auto selects the number of components to capture
            at least this fraction of total variance. Overrides `n_components`.
        max_components : int, optional
            Maximum number of components to consider when computing cumulative
            explained variance. Default is 100.

    Returns:
        cp.ndarray: Orthogonal transformation matrix that aligns `data` to `reference`.

    Notes
    -----
    - When using `truncated` SVD and `variance_threshold`, an initial fit with 
      `max_components` is used to estimate explained variance.
    - Assumes timepoints / targets are along the first dimension (rows).
    """

    if not isinstance(variance_threshold, cp.ndarray):
        variance_threshold = cp.asarray(variance_threshold)

    X = cp.matmul(data.T, reference)

    # Full SVD
    if svd_mode == "full":
        U, S, Vt = cp.linalg.svd(X, full_matrices=False)
    
    # Truncated SVD
    elif svd_mode == "truncated":
        # Fit with max_components to estimate explained variance
        _, S_temp, _ = tl.tenalg.svd_interface(X, method="truncated_svd", 
                                                n_eigenvecs=max_components,
                                                n_iter_mask_imputation=5)
        
        total_variance = cp.sum(S_temp ** 2)

        if total_variance == 0 or cp.isnan(total_variance):
            return cp.eye(X.shape[0], X.shape[1])
        
        explained_variance_ratio = (S_temp ** 2) / cp.sum(S_temp ** 2)
        cumulative_variance = cp.cumsum(explained_variance_ratio)
        n_components = min(cp.searchsorted(cumulative_variance, variance_threshold) + 1,
                        len(cumulative_variance))
        variance_captured = cumulative_variance[n_components-1].item()

        if cumulative_variance[-1] < variance_threshold:
            print(f"Warning: Only {cumulative_variance[-1]:.4f} variance explained with max_components={max_components}.")

        # print('Final number of components used for SVD:', n_components,
        #         'explains {:.4f} of variance.'.format(variance_captured))
        
        # Refit with final n_components
        U, _, Vt = tl.tenalg.svd_interface(X, method="truncated_svd", 
                                    n_eigenvecs=n_components,
                                    n_iter_mask_imputation=5)

    return cp.matmul(U, Vt)

##### TEST #####
# svd_mode = "truncated"
# n_components = 4
# variance_threshold = 0.7
# select_vox = 200

# transmat_gpu = calculate_transmat(
#     normalized_data_gpu[0][:, :select_vox],
#     normalized_data_gpu[1][:, :select_vox],
#     svd_mode=svd_mode,
#     n_components=n_components,
#     variance_threshold=cp.array(variance_threshold)
# )

# from hyperalign_utils import calculate_transmat
# transmat_cpu = calculate_transmat(
#     cp.asnumpy(normalized_data_gpu[0][:, :select_vox]),
#     cp.asnumpy(normalized_data_gpu[1][:, :select_vox]),
#     svd_mode=svd_mode,
#     n_components=n_components,
#     variance_threshold=variance_threshold
# )

# diff = transmat_gpu.flatten().get() - transmat_cpu.flatten()
# print(diff.min())
# print(diff.max())
# # assert np.allclose(transmat_cpu, cp.asnumpy(transmat_gpu)), "GPU and CPU have different results"

# import matplotlib.pyplot as plt 
# plt.imshow(transmat_gpu.get()-transmat_cpu)
# plt.colorbar()
# plt.show()

# gpu_aligned = normalized_data_gpu[0][:, :select_vox] @ transmat_gpu
# cpu_aligned = normalized_data_gpu[0][:, :select_vox].get() @ transmat_cpu
# diff = gpu_aligned.get() - cpu_aligned 
# plt.imshow(gpu_aligned.get() - cpu_aligned)
# plt.colorbar()
# plt.show()

# plt.plot(diff.flatten())
# plt.show()

#%%
def calculate_isc(data_batch, isc_type="row", ref_subject="common", fisher_z=False):
    """
    Computes Inter-Subject Correlation (ISC) for fMRI time-series.

    Parameters
        data_batch (cp.ndarray): A 3D CuPy array of shape (n_subjects, n_timepoints/n_targets/n_voxels, n_voxels).
        isc_type (str): Direction of correlation: "row" or "column". Default is 'row'.
        ref_subject (str): Reference against which each subject is correlated:
            - 'common': Use leave-one-out average of other subjects (default).
            - int: Index of a specific subject to be used as fixed reference.
        fisher_z (bool): If True, applies Fisher z-transform (arctanh) to correlation values.

    Returns
        cp.ndarray: 1D CuPy array of ISC values.
    """

    valid_types = {"column", "row"}
    if isc_type not in valid_types:
        raise ValueError(f"isc_type must be one of {valid_types}")

    n_subjects = data_batch.shape[0]
    isc = cp.empty(n_subjects, dtype=cp.float32)

    if isinstance(ref_subject, str) and ref_subject == "common":
        # Use leave-one-out average of other subjects
        for i in range(n_subjects):
            others_idx = cp.arange(n_subjects) != i
            others = data_batch[others_idx]  # shape (n_subjects-1, n_timepoints, n_voxels)

            if others.shape[0] == 1:
                ref_data = others[0]
            else:
                ref_data = cp.mean(others, axis=0)  # mean over subjects; shape (n_timepoints, n_voxels)

            subject_data = data_batch[i]  # shape (n_timepoints, n_voxels)
            isc_vals = calculate_matrix_correlation(subject_data, ref_data, mode=isc_type)
            isc[i] = cp.nanmean(isc_vals)
    else:
        # Fixed reference subject (integer index)
        ref_data = data_batch[ref_subject]
        for i in range(n_subjects):
            if i == ref_subject:
                isc[i] = 0.99 if fisher_z else 1
            else:
                subject_data = data_batch[i]
                isc_vals = calculate_matrix_correlation(subject_data, ref_data, mode=isc_type)
                isc[i] = cp.nanmean(isc_vals)

    if fisher_z:
        isc = cp.arctanh(isc)

    return isc

#### TEST ####
# isc_type = "column"
# ref_subject = 0
# ref_subject = cp.array(0)
# fisher_z=False


# def calculate_matrix_correlation(data1, data2, mode):
#     axis = 0 if mode == "column" else 1
    
#     data1_mean = cp.mean(data1, axis=axis, keepdims=True)
#     data2_mean = cp.mean(data2, axis=axis, keepdims=True)

#     data1_demeaned = data1 - data1_mean
#     data2_demeaned = data2 - data2_mean

#     numerator = cp.sum(data1_demeaned * data2_demeaned, axis=axis)
#     denominator = cp.sqrt(cp.sum(data1_demeaned ** 2, axis=axis) * cp.sum(data2_demeaned ** 2, axis=axis))

#     correlations = numerator / denominator
#     correlations = cp.nan_to_num(correlations, nan=0.0)

#     return correlations

# isc_gpu = calculate_isc(normalized_data_gpu, isc_type, ref_subject, fisher_z)

# from hyperalign_utils import calculate_isc
# isc_cpu = calculate_isc(normalized_data_cpu, isc_type, ref_subject.get(), fisher_z)

# isc_cpu_array = cp.asarray(isc_cpu)
# assert cp.allclose(isc_gpu, isc_cpu_array, atol=1e-6), "GPU and CPU results differ"

# print(isc_cpu)
# print(isc_gpu)

#%%
def calculate_isc_rep_geometry(data_batch, ref_subject="common", fisher_z=True):
    """
    Computes Representational Geometry Inter-Subject Correlation (ISC) on GPU.

    This ISC variant measures the similarity of representational geometries between
    subjects, where geometry is defined by the pairwise correlation structure between voxels.

    Parameters
        data_batch : cp.ndarray
            A 3D CuPy array of shape (n_subjects, n_timepoints, n_voxels).
        ref_subject : str or int, optional
            - 'common': Use leave-one-out average of other subjects' geometry matrices (default).
            - int: Use a fixed subject index as the reference geometry.
        fisher_z : bool, optional
            Whether to apply Fisher z-transform (arctanh) to the final ISC values. Default is True.

    Returns
        cp.ndarray: A 1D CuPy array of ISC values.

    Notes
    -----
    - ISC is calculated as the Pearson correlation between the upper triangles
      (excluding the diagonal) of each subject's voxel-voxel correlation matrix
      and the corresponding reference matrix.
    """

    n_subjects, n_timepoints, n_voxels = data_batch.shape

    # Create empty correlation matrices for each subject
    targets = cp.empty((n_subjects, n_voxels, n_voxels), dtype=cp.float32)

    for i in range(n_subjects):
        corr = cp.corrcoef(data_batch[i], rowvar=False)  # shape (n_vox, n_vox)
        targets[i] = corr
    
    # Create empty isc array: 
    isc = cp.empty(n_subjects)

    # Determine upper triangle indices (excluding diagonal)
    triu_indices = cp.triu_indices(n_voxels, k=1)


    for i in range(n_subjects):
        target = targets[i]
        if isinstance(ref_subject, str) and ref_subject == "common":
            # Leave-one-out mean reference
            others = cp.stack([targets[j] for j in range(n_subjects) if j != i], axis=0)
            ref = cp.mean(others, axis=0)
        else:
            # Fixed reference subject
            if i == ref_subject:
                # Correlation of subject with itself = 1
                isc[i] = 1.0
                continue
            
            ref = targets[ref_subject]

        # Vectorize upper triangles
        ref_vectorized = ref[triu_indices]
        target_vectorized = target[triu_indices]

        # Compute ISC as correlation between vectorized upper triangles
        mask = ~cp.isnan(ref_vectorized) & ~cp.isnan(target_vectorized)
        isc[i] = cp.corrcoef(ref_vectorized[mask], target_vectorized[mask])[0, 1]

    if fisher_z:
        isc = cp.clip(isc, -0.99, 0.99)
        isc = cp.arctanh(isc)
        
    return isc

### TEST ###
# ref_subject = 0
# fisher_z=True
# isc_gpu = calculate_isc_rep_geometry(normalized_data_gpu[:, :, :select_vox], ref_subject, fisher_z)

# from hyperalign_utils import calculate_isc_rep_geometry
# isc_cpu = calculate_isc_rep_geometry([data[:, :select_vox] for data in normalized_data_cpu], ref_subject, fisher_z)
# isc_cpu_array = cp.asarray(isc_cpu)

# assert cp.allclose(isc_gpu, isc_cpu_array, atol=1e-6), "GPU and CPU results differ"
# print(isc_cpu)
# print(isc_gpu)

#%%
def calculate_disparity_matrix(data_batch):
    """
    Computes pairwise Frobenius norm disparities between all subjects' data.

    Parameters
        data_batch : cp.ndarray
            A 3D CuPy array of shape (n_subjects, timepoints/targets/voxels, voxels)

    Returns
        cp.ndarray: A 2D CuPy array of shape (n_subjects, n_subjects) with pairwise disparities between subjects.

    Notes
    -----
    - The Frobenius norm is computed over the difference between each pair of subject matrices.
    - Result is symmetric with zeros on the diagonal.
    """
    n_subjects = data_batch.shape[0]
    
    # Compute pairwise differences
    diffs = data_batch[:, None, :, :] - data_batch[None, :, :, :]  # shape: (n_subjects, n_subjects, T, V)
    
    # Compute Frobenius norms along last two dims
    disparity_mat = cp.linalg.norm(diffs.reshape(n_subjects, n_subjects, -1), axis=2)
    
    return disparity_mat

##### TEST #####
# disparity_gpu = calculate_disparity_matrix(normalized_data_gpu)

# from hyperalign_utils import calculate_disparity_matrix
# disparity_cpu = calculate_disparity_matrix(normalized_data_cpu)
# disparity_cpu_array = cp.asarray(disparity_cpu)

# assert cp.allclose(disparity_gpu, disparity_cpu_array, atol=1e-6), "GPU and CPU results differ"

# print(disparity_cpu)
# print(disparity_gpu)

#%%
def define_targets(data_batch, atlas, brain_mask):
    """
    Computes regional target time series by averaging voxel activity within atlas-defined regions,
    restricted to brain-masked voxels.

    Parameters
        data_batch : array
            A 3D array of shape (n_subjects, n_timepoints, n_voxels).
        atlas : array
            A 1D or 3D array with integer region labels. If 3D, it is flattened internally.
        brain_mask : array
            A binary (0 or 1) array of the same shape as `atlas` indicating which voxels
            should be included in the analysis. If 3D, it is also flattened internally.

    Returns
        array
            A 3D array of shape (n_subjects, n_timepoints, n_regions), where each element
            represents the average signal across voxels within a given region for each subject
            and timepoint.

    Notes
    -----
    - Voxels labeled as 0 in the atlas are treated as background and ignored.
    - Regions with no voxels remaining after masking are filled with `NaN`.
    - Ensure voxel dimensions in the data match the flattened brain mask and atlas.
    """

    if atlas.ndim == 3:
        atlas = atlas.flatten()
    if brain_mask.ndim == 3:
        brain_mask = brain_mask.flatten()

    if atlas.shape != brain_mask.shape:
        raise ValueError(f"atlas and brain_mask must have the same shape, got {atlas.shape} and {brain_mask.shape}")

    # Apply mask to atlas to get masked region labels
    atlas_masked = atlas[brain_mask == 1]

    # Get unique regions excluding background (0)
    regions = np.unique(atlas_masked)
    regions = regions[regions != 0]
    n_regions = len(regions)
    if n_regions == 0:
        raise ValueError("No valid regions found in the atlas after masking.")

    n_subjects, n_timepoints, _ = data_batch.shape

    # Preallocate output: subjects x timepoints x regions
    target_timeseries = np.empty((n_subjects, n_timepoints, n_regions), dtype=cp.float32)

    for i, region_label in enumerate(regions):
        region_voxel_indices = (atlas_masked == region_label)  # shape: (n_masked_voxels,)
        if not np.any(region_voxel_indices):
            target_timeseries[:, :, i] = np.nan 
            continue

        # Extract data for region: (n_subjects, n_timepoints, n_region_voxels)
        region_data = data_batch[:, :, region_voxel_indices]

        # Average across voxels (last axis)
        region_mean = np.mean(region_data, axis=2)

        # Store
        target_timeseries[:, :, i] = region_mean

    return target_timeseries


##### TEST #####
# import os 
# base_path = "/media/tram/Elements/nsd_tram/natural-scenes-dataset"
# # base_path =r"C:\Users\Tram\Desktop\data"
# input_path = os.path.join(base_path, "nsddata_preproc")
# template_dir = os.path.join(base_path, "nsddata", "templates" )

# atlas = cp.array(np.load(os.path.join(template_dir,"harvard_oxford_atlas_5mm.npy")))
# targets_gpu = define_targets(masked_gpu_data, atlas, mni_brain_mask)

# from hyperalign_utils import define_targets
# targets_cpu = define_targets(masked_cpu_data, atlas.get(), mni_brain_mask.get())
# targets_cpu_array = cp.asarray(targets_cpu)

# assert cp.allclose(targets_gpu, targets_cpu, atol=1e-6), "GPU and CPU results differ"

# diff = targets_gpu - targets_cpu_array
# print(diff.max())
# print(diff.min())
# plt.plot(diff.get().flatten())

#%%
def get_searchlight_indices(center, image_shape, radius):
    """
    Returns the voxel indices within a cubic searchlight centered at a given point,
    constrained to lie within the bounds of a brain volume.

    Parameters
        center : array-like (length 3)
            The (x, y, z) coordinates of the searchlight center.
        image_shape : tuple
            Shape of the 4D fMRI volume (x, y, z, timepoints). Only the spatial
            dimensions (first three) are used to constrain the searchlight region.
        radius : int or cp.ndarray
            Radius of the searchlight in voxel units. A radius of 1 yields a 3x3x3 cube.

    Returns
        cp.ndarray
            A 2D CuPy array of shape (n_points, 3), where each row contains the (x, y, z)
            coordinates of a voxel within the searchlight cube and within volume bounds.

    Notes
    - The cube includes the center voxel and all neighbors within the specified radius.
    """

    if center.shape != (3,) or not cp.all(cp.isfinite(center)):
        raise ValueError("center must be a 3-element array of finite values")
    
    # Convert radius to scalar if CuPy array
    if isinstance(radius, cp.ndarray):
        radius = radius.item()

    # Define volume bounds from image shape
    min_coords = cp.array([0, 0, 0])
    max_coords = cp.array([s - 1 for s in image_shape[:3]])

    # Generate ranges for each dimension
    x_range = cp.arange(center[0].item() - radius, center[0].item() + radius + 1)
    y_range = cp.arange(center[1].item() - radius, center[1].item() + radius + 1)
    z_range = cp.arange(center[2].item() - radius, center[2].item() + radius + 1)
    
    # Create meshgrid of all points
    x, y, z = cp.meshgrid(x_range, y_range, z_range, indexing="ij")
    
    # Stack coordinates into a single array
    all_points = cp.column_stack((x.ravel(), y.ravel(), z.ravel()))
    
    # Create mask for points within volume bounds
    within_bounds = cp.all((all_points >= min_coords) & (all_points <= max_coords), axis=1)
    valid_points = all_points[within_bounds]

    return valid_points.astype(cp.int32)


##### TEST #####
# center = cp.array([2,10,20])
# image_shape = all_data_gpu_4D[0].shape
# radius = cp.array(5)

# all_points_gpu = get_searchlight_indices(center, image_shape, radius)

# from hyperalign_utils import get_searchlight_indices
# all_points_cpu = get_searchlight_indices(center.get(), image_shape, int(radius.get()))

# assert np.all(all_points_gpu.get() == all_points_cpu), "GPU and CPU results differ"
# print(all_points_cpu)
# print(all_points_gpu)

#%%
def voxel_tracking(input, image_shape, conversion_type):
    """
    Converts between flattened voxel indices and 3D voxel coordinates in anatomical (x, y, z) space.

    Parameters
        input : int, tuple, or cp.ndarray
            Input for conversion:
            - If conversion_type is "index_to_coord": a scalar index (int or 0-D CuPy array).
            - If conversion_type is "coord_to_index": a tuple or 1D CuPy array of (x, y, z) voxel coordinates.
        image_shape : tuple
            The shape of the 3D brain volume (x_dim, y_dim, z_dim). Only the first three dimensions are used.
        conversion_type : str
            Type of conversion to perform:
            - "index_to_coord": Converts a flat voxel index into a 3D coordinate.
            - "coord_to_index": Converts a 3D coordinate into a flat voxel index.

    Returns
        cp.ndarray or int
            - If "index_to_coord": returns a CuPy array of shape (3,) representing (x, y, z).
            - If "coord_to_index": returns a single integer representing the flat voxel index.
    """

    x_dim, y_dim, z_dim = image_shape[:3]
    
    if conversion_type == "index_to_coord":
        # Ensure input is integer scalar (cupy scalar or int)
        idx = int(input) if isinstance(input, (int, cp.integer)) else int(input.item())
        z = idx % z_dim
        y = (idx // z_dim) % y_dim
        x = idx // (y_dim * z_dim)
        return cp.array([x, y, z])
    
    else:  # coord_to_index
        x, y, z = input
        return int(x) * (y_dim * z_dim) + int(y) * z_dim + int(z)

##### TEST #####
# input = cp.array(6)
# conversion_type = "index_to_coord"
# input = cp.array ([2,3,4])
# conversion_type = "coord_to_index"

# results_gpu = voxel_tracking(input, image_shape=all_data_gpu_4D[0].shape, conversion_type=conversion_type)

# from hyperalign_utils import voxel_tracking
# # input = int(input)
# input = input.get()
# results_cpu = voxel_tracking(input, image_shape=all_data_gpu_4D[0].shape, conversion_type=conversion_type)

# print(results_gpu)
# print(results_cpu)

#%%
def create_index_mapping(mask):
    """
    Creates a lookup table that maps original voxel indices to new indices
    after masking, with excluded (masked-out) voxels assigned -1.

    Parameters
    ----------
    mask : cp.ndarray
        A 1D or flattenable CuPy array of binary values (0 or 1),
        where 1 indicates voxels to include.

    Returns
    -------
    cp.ndarray
        A 1D CuPy array of the same length as `mask`, where:
        - Included voxel indices are mapped to consecutive integers starting from 0.
        - Excluded voxel indices are set to -1.

    """
    
    if len(mask.shape) != 1:
        mask = mask.flatten()

    index_mapping = cp.full(mask.shape, -1, dtype=cp.int32)

    new_index = 0
    for old_index in range(mask.size):
        if mask[old_index] == 1:
            index_mapping[old_index] = new_index
            new_index += 1

    return index_mapping

##### TEST #####
# index_mapping_gpu = create_index_mapping(mni_brain_mask)

# from hyperalign_utils import create_index_mapping
# index_mapping_cpu = create_index_mapping(mni_brain_mask)

# index_mapping_cpu = np.array([-1 if v is None else v for v in index_mapping_cpu.values()])
# assert np.all(index_mapping_cpu == index_mapping_gpu.get()), "GPU and CPU have diff results"


#%%
def map_indices(old_indices, index_array):
    """
    Map old_indices to new indices using index_array.
    
    Parameters:
        old_indices: cp.ndarray of old indices (ints)
        index_array: cp.ndarray, mapping old_index -> new_index or -1
    
    Returns:
        cp.ndarray of mapped indices (or -1 if missing)
    """
    return index_array[old_indices]

##### TEST #####
# old_indices = cp.array([56842,5000,72000])
# new_indices = map_indices(old_indices, index_mapping_gpu)
# print(new_indices)

# from hyperalign_utils import map_indices
# new_indices = map_indices(old_indices.get(), index_mapping_cpu)
# print(new_indices)


#%%
def clean_gpu_memory(keep_vars=None):
    """
    Deletes all CuPy arrays from global scope except those in keep_vars.
    Also triggers garbage collection and frees CuPy's memory pool.
    
    Parameters:
    - keep_vars: list of variable names (strings) to keep (optional).
    """
    if keep_vars is None:
        keep_vars = []

    current_globals = globals()
    
    # Find and delete all CuPy arrays except those in keep_vars
    for var_name in list(current_globals):
        if var_name in keep_vars:
            continue
        var_val = current_globals[var_name]
        if isinstance(var_val, cp.ndarray):
            del current_globals[var_name]

    # Garbage collect and clear memory pool
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()



    
