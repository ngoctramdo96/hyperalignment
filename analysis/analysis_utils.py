import numpy as np
import h5py
import nibabel as nib
from nilearn import plotting
import os

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


def calculate_matrix_correlation(data1, data2, mode):
    """
    Calculates the correlation between corresponding rows or columns of two matrices.

    Parameters:
        data1, data2 (np.ndarray): Input matrices.
        mode (str): Use "column" to compute correlation between corresponding columns,
                    or "row" to compute between corresponding rows.

    Returns:
        np.ndarray: 1D array of correlation coefficients between the corresponding
                    rows or columns of `data1` and `data2`.
    """
    if mode == "column":
        axis = 0
    elif mode == "row":
        axis = 1
    else:
        raise ValueError("mode must be column or row")
    
    # Demean
    data1_mean = np.mean(data1, axis=axis, keepdims=True)
    data2_mean = np.mean(data2, axis=axis, keepdims=True)
    data1_demeaned = data1 - data1_mean
    data2_demeaned = data2 - data2_mean

    # Calculate correlation
    numerator = np.sum(data1_demeaned * data2_demeaned, axis=axis)
    denominator = np.sqrt(np.sum(data1_demeaned ** 2, axis=axis) * np.sum(data2_demeaned ** 2, axis=axis))

    correlations = numerator / denominator
    # correlations = np.nan_to_num(correlations, nan=0.0)

    return correlations

def load_h5py_data(file_path):
    data = {}
    with h5py.File(file_path, 'r') as f:
        def recursive_load(name, obj):
            if isinstance(obj, h5py.Dataset):
                if obj.shape == ():  # scalar dataset
                    data[name] = obj[()]
                else:
                    data[name] = obj[:]
        
        f.visititems(recursive_load)

    if len(data) == 1:
        return next(iter(data.values()))
    else:
        return data

def calculate_isc_matrix(data_batch, isc_type="row", ref_subject="common", fisher_z=False):
    """
    """
    
    valid_types = {"column", "row"}
    if isc_type not in valid_types:
        raise ValueError(f"isc_type must be one of {valid_types}")

    n_subject, n_t, n_voxel = data_batch.shape
    
    if isc_type == "row":
        isc = np.empty((n_subject, n_t), dtype=np.float32)
    elif isc_type == "column":
        isc = np.empty((n_subject, n_voxel), dtype=np.float32)

    if isinstance(ref_subject, str) and ref_subject == "common":
        # Use leave-one-out average of other subjects
        for i in range(n_subject):
            others_idx = np.arange(n_subject) != i
            others = data_batch[others_idx]  # shape (n_subjects-1, n_timepoints/targets, n_voxels)

            if others.shape[0] == 1:
                ref_data = others[0]
            else:
                ref_data = np.mean(others, axis=0)  # mean over subjects; shape (n_timepoints, n_voxels)

            subject_data = data_batch[i]  # shape (n_timepoints, n_voxels)
            isc_vals = calculate_matrix_correlation(subject_data, ref_data, mode=isc_type)
            isc[i, :] = isc_vals
    else:
        # Fixed reference subject (integer index)
        ref_data = data_batch[ref_subject]
        for i in range(n_subject):
            if i == ref_subject:
                isc[i] = 0.99 if fisher_z else 1
            else:
                subject_data = data_batch[i]
                isc_vals = calculate_matrix_correlation(subject_data, ref_data, mode=isc_type)
                isc[i, :] = isc_vals

    if fisher_z:
        isc = np.arctanh(isc)

    return isc


def calculate_isc(data_batch, isc_type="row", ref_subject="common", fisher_z=False):
    """
    Computes Inter-Subject Correlation (ISC) for fMRI time-series.

    Parameters
        data_batch (np.ndarray): A 3D array of shape (n_subjects, n_timepoints/n_targets/n_voxels, n_voxels).
        isc_type (str): Direction of correlation: "row" or "column". Default is 'row'.
        ref_subject (str): Reference against which each subject is correlated:
            - 'common': Use leave-one-out average of other subjects (default).
            - int: Index of a specific subject to be used as fixed reference.
        fisher_z (bool): If True, applies Fisher z-transform (arctanh) to correlation values.

    Returns
        np.ndarray: 1D array of ISC values.
    """

    valid_types = {"column", "row"}
    if isc_type not in valid_types:
        raise ValueError(f"isc_type must be one of {valid_types}")

    n_subjects = data_batch.shape[0]
    isc = np.empty(n_subjects, dtype=np.float32)

    if isinstance(ref_subject, str) and ref_subject == "common":
        # Use leave-one-out average of other subjects
        for i in range(n_subjects):
            others_idx = np.arange(n_subjects) != i
            others = data_batch[others_idx]  # shape (n_subjects-1, n_timepoints, n_voxels)

            if others.shape[0] == 1:
                ref_data = others[0]
            else:
                ref_data = np.mean(others, axis=0)  # mean over subjects; shape (n_timepoints, n_voxels)

            subject_data = data_batch[i]  # shape (n_timepoints, n_voxels)
            isc_vals = calculate_matrix_correlation(subject_data, ref_data, mode=isc_type)
            isc[i] = np.nanmean(isc_vals)
    else:
        # Fixed reference subject (integer index)
        ref_data = data_batch[ref_subject]
        for i in range(n_subjects):
            if i == ref_subject:
                isc[i] = 0.99 if fisher_z else 1
            else:
                subject_data = data_batch[i]
                isc_vals = calculate_matrix_correlation(subject_data, ref_data, mode=isc_type)
                isc[i] = np.nanmean(isc_vals)

    if fisher_z:
        isc = np.arctanh(isc)

    return isc


def reconstruct_img(data_2D, img_shape, affine,
                    brain_mask, save_nifti=False, save_path=None):
    """

    """
    # Initialize the full 2D image with non-brain voxels excluded previously
    if len(img_shape) == 3:
        x,y,z = img_shape
        t = 1
    elif len(img_shape) == 4:
        x,y,z,t = img_shape
        
    data_2D_full = np.zeros(shape = (t, x*y*z))

    # Add axis in case data_2D is 1 dim
    if len(data_2D.shape) == 1:
        data_2D = data_2D[np.newaxis, :]

    # prep brain mask 
    brain_mask = brain_mask.flatten()
    brain_voxels = np.where(brain_mask == 1)[0]  # Get indices of valid voxels
    
    # Fill the full 2D image with the hyperaligned data
    for data_index, voxel_index in enumerate(brain_voxels):
        data_2D_full[:, voxel_index] = data_2D[:, data_index]
    
    # Convert 2D to 4D
    reconstructed_img = data_2D_full.T.reshape(img_shape)
    reconstructed_img = np.squeeze(reconstructed_img)
    reconstructed_nifti_img = nib.Nifti1Image(reconstructed_img, affine=affine)

    if save_nifti:
        nib.save(reconstructed_nifti_img, save_path)

    return reconstructed_nifti_img

def reconstruct_img_fr_region(data_region, img_shape, affine, atlas, save_nifti=False, save_path=None):
    """
    """
    # Initialize the full 2D image with non-brain voxels excluded previously
    if len(img_shape) == 3:
        x,y,z = img_shape
        t = 1
    elif len(img_shape) == 4:
        x,y,z,t = img_shape
        
    data_2D_full = np.zeros(shape = (t, x*y*z))

    # Add axis in case data_2D is 1 dim
    if len(data_region.shape) == 1:
        data_region = data_region[np.newaxis, :]

    atlas = atlas.flatten()
    
    # Fill the full 2D image with the hyperaligned data
    for i, region_label in enumerate(np.unique(atlas)[1:]): #skip Background
        region_voxel_indices = np.where(atlas == region_label)[0]
        data_2D_full[:, region_voxel_indices] = data_region[:, i][:, np.newaxis]
    
    # Convert 2D to 4D
    reconstructed_img = data_2D_full.T.reshape(img_shape)
    reconstructed_img = np.squeeze(reconstructed_img)
    reconstructed_nifti_img = nib.Nifti1Image(reconstructed_img, affine=affine)

    if save_nifti:
        nib.save(reconstructed_nifti_img, save_path)

    return reconstructed_nifti_img


def save_interactive_viewers(images, titles, filename, cmap='jet', bg_img='MNI152',
                             threshold=1e-06, vmin=None, vmax=None):
    """
    Save an interactive HTML file with nilearn.view_img viewers for multiple images
    arranged in 2 columns.

    Parameters
    ----------
    images : list of Nifti-like images to display
    titles : list of section titles for each image
    vmin : float, optional, min value for colormap scaling
    vmax : float, default=0.5, max value for colormap scaling
    cmap : str, default='jet', colormap to use
    """

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*partition.*MaskedArray.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*created more than 10 nilearn views.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Non-finite values detected.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*vmin cannot be chosen when cmap is symmetric*")


    # Create viewers
    viewers = [
        plotting.view_img(img, cmap=cmap, bg_img=bg_img, threshold=threshold,
                          cut_coords=(0, 0, 0), vmin=vmin, vmax=vmax)

        for img in images
    ]

    # Build HTML content in a 2-column layout
    sections = "\n".join(
        f"<div class='viewer'><h2>{title}</h2>{view._repr_html_()}</div>"
        for title, view in zip(titles, viewers)
    )

    html_content = f"""
    <html>
      <head>
        <title>Interactive Viewer</title>
        <style>
          body {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
          }}
          .viewer {{
            flex: 1 1 calc(50% - 20px); /* 2 columns */
            box-sizing: border-box;
          }}
          .viewer h2 {{
            text-align: center;
          }}
        </style>
      </head>
      <body>
        {sections}
      </body>
    </html>
    """

    # Write to file
    with open(filename, "w") as f:
        f.write(html_content)

    print(f"Interactive HTML saved: {filename}")



def calculate_isc_rep_geometry(data_batch, ref_subject="common", fisher_z=True, cal_corr=True):
    n_subjects, n_t, n_voxels = data_batch.shape

    # Create empty correlation matrices for each subject
    targets = np.empty((n_subjects, n_voxels, n_voxels), dtype=np.float32)

    for i in range(n_subjects):
        if cal_corr: 
            corr = np.corrcoef(data_batch[i], rowvar=False)  # shape (n_vox, n_vox)
            targets[i] = corr
        else:
            targets[i] = data_batch[i] #case where input are voxel x voxel or region x region already 
            
    # Create empty isc array: 
    isc = np.empty(n_subjects)

    # Determine upper triangle indices (excluding diagonal)
    triu_indices = np.triu_indices(n_voxels, k=1)


    for i in range(n_subjects):
        target = targets[i]
        if isinstance(ref_subject, str) and ref_subject == "common":
            # Leave-one-out mean reference
            others = np.stack([targets[j] for j in range(n_subjects) if j != i], axis=0)
            ref = np.mean(others, axis=0)
        else:
            # Fixed reference
            if i == ref_subject:
                isc[i] = 1.0
                continue

            ref = targets[ref_subject]

        # Vectorize upper triangles
        ref_vectorized = ref[triu_indices]
        target_vectorized = target[triu_indices]
        del target, ref

        mask = ~np.isnan(ref_vectorized) & ~np.isnan(target_vectorized)
        if np.sum(mask) < 2:   # need at least 2 values
            isc[i] = np.nan
        else:
            isc[i] = np.corrcoef(ref_vectorized[mask], target_vectorized[mask])[0, 1]

    if fisher_z:
        isc = np.clip(isc, -0.99, 0.99)
        isc = np.arctanh(isc)
        
    return isc


def load_and_align(beta, path, subj_str, kind, reference):
    """Helper to load transformation matrix and align beta map."""
    fname = f"R_subj{subj_str}_parcellation_ref-{reference}.h5" if "fc" in kind else f"R_subj{subj_str}_spatial_ref-{reference}.h5"
    file = os.path.join(path, fname)
    R = load_h5py_data(file)
    return beta @ R

def region_average(data_batch, atlas, brain_mask):

    if atlas.ndim == 3:
        atlas = atlas.flatten()
    if brain_mask.ndim == 3:
        brain_mask = brain_mask.flatten()

    if atlas.shape != brain_mask.shape:
        raise ValueError(f"atlas and brain_mask must have the same shape, got {atlas.shape} and {brain_mask.shape}")

    # Masked atlas (ignore background=0 and outside brain)
    atlas_masked = atlas[brain_mask == 1]

    # Get unique regions excluding background
    regions = np.unique(atlas_masked)
    regions = regions[regions != 0]
    if len(regions) == 0:
        raise ValueError("No valid regions found in the atlas after masking.")

    n_subjects, n_timepoints, n_voxels = data_batch.shape
    n_regions = len(regions)

    # Preallocate output
    region_timeseries = np.zeros((n_subjects, n_timepoints, n_regions))

    for i, region_label in enumerate(regions):
        region_voxel_indices = np.where(atlas_masked == region_label)[0]
        if len(region_voxel_indices) == 0:
            continue

        # Extract data for region: (n_subjects, n_timepoints, n_region_voxels)
        region_data = data_batch[:, :, region_voxel_indices]

        # Average across voxels (last axis) → (n_subjects, n_timepoints)
        region_mean = np.mean(region_data, axis=2)

        # Assign mean back to all voxels in the region
        region_timeseries[:, :, i] = region_mean

    return region_timeseries

# exploring region with top isc
def top_values(arr, min_value=0):
    # Find indices of values above the threshold
    top_indices = np.where(arr > min_value)[0]
    top_values = arr[top_indices]
    
    if len(top_values) == 0:
        return np.array([]), np.array([])  # No positive values above threshold
    
    # Sort values in descending order
    sorted_indices = np.argsort(top_values)[::-1]
    
    # Map back to original array indices
    top_indices_ori = top_indices[sorted_indices]
    top_values_ori = arr[top_indices]
    
    return top_values_ori, top_indices_ori

# def region_average(data_batch, atlas, brain_mask):
#     """
#     Computes voxelwise target time series where each voxel's value is replaced by
#     the mean time series of the region it belongs to (based on atlas),
#     restricted to brain-masked voxels.

#     Parameters
#         data_batch : array
#             A 3D array of shape (n_subjects, n_timepoints, n_voxels).
#         atlas : array
#             A 1D or 3D array with integer region labels. If 3D, it is flattened internally.
#         brain_mask : array
#             A binary (0 or 1) array of the same shape as `atlas` indicating which voxels
#             should be included in the analysis. If 3D, it is also flattened internally.

#     Returns
#         array
#             A 3D array of shape (n_subjects, n_timepoints, n_voxels), where each voxel
#             has been replaced with the average signal of its region.
#     """

#     if atlas.ndim == 3:
#         atlas = atlas.flatten()
#     if brain_mask.ndim == 3:
#         brain_mask = brain_mask.flatten()

#     if atlas.shape != brain_mask.shape:
#         raise ValueError(f"atlas and brain_mask must have the same shape, got {atlas.shape} and {brain_mask.shape}")

#     # Masked atlas (ignore background=0 and outside brain)
#     atlas_masked = atlas[brain_mask == 1]

#     # Get unique regions excluding background
#     regions = np.unique(atlas_masked)
#     regions = regions[regions != 0]
#     if len(regions) == 0:
#         raise ValueError("No valid regions found in the atlas after masking.")

#     n_subjects, n_timepoints, n_voxels = data_batch.shape

#     # Preallocate output
#     region_timeseries = np.zeros((n_subjects, n_timepoints, n_voxels))

#     for region_label in regions:
#         region_voxel_indices = np.where(atlas_masked == region_label)[0]
#         if len(region_voxel_indices) == 0:
#             continue

#         # Extract data for region: (n_subjects, n_timepoints, n_region_voxels)
#         region_data = data_batch[:, :, region_voxel_indices]

#         # Average across voxels (last axis) → (n_subjects, n_timepoints)
#         region_mean = np.mean(region_data, axis=2)

#         # Assign mean back to all voxels in the region
#         region_timeseries[:, :, region_voxel_indices] = region_mean[:, :, None]

#     return region_timeseries



#### TEST ####
# import numpy as np
# data_4D = np.random.rand(2, 2, 3, 2)  # Example 4D data
# reshaped_data = data_4D.reshape(-1, 2).T

# brain_mask = np.ones(shape=(2,2,3))
# brain_mask[0, 0, 0] = 0
# brain_mask[1, 0, 1] = 0
# brain_mask_flattened = brain_mask.flatten()

# reshaped_data[:, np.where(brain_mask_flattened==0)]=0
# reshaped_data_no_zeros = reshaped_data[:, ~np.all(reshaped_data == 0, axis=0)]
# reshaped_data_no_zeros.shape

# data_4D_reconstructed = reconstruct_img(reshaped_data_no_zeros, original_shape=data_4D.shape, affine=None,
#                     brain_mask=brain_mask, save_nifti=False, save_path=None)

# np.all(data_4D_reconstructed == data_4D)

