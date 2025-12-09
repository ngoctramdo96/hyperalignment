# Hyperalignment

## Goals
The main goals of this project are:
* Implement two types of **hyperalignment** in a **searchlight manner**, following the methodology described in Guntupalli et al. (2016):
    * Response-based Hyperalignment (RHA)
    * Connectivity-based Hyperalignment (CHA)

* Test **cross-modal application**, specifically evaluating the ability to use transformation matrices acquired from resting-state data to predict task-based data

Unlike common implementations that apply hyperalignment on surface-based data, this project focuses on **voxel-based** data to enable broader application in volumetric fMRI analysis pipelines.


## What is Hyperalignment?
Hyperalignment, introduced by Haxby et al. (2011), is a method that aligns individual patterns of brain activity into a shared, common information space. Although individuals may be exposed to the same stimuli—such as viewing the same image or experiencing a similar mental state like relaxation—the resulting neural signals can differ due to unique functional topographies, i.e., differences in how each brain organizes information across the anatomical space measured by fMRI.

Hyperalignment addresses this by finding transformations that map each individual's data into a common representational space, preserving fine-grained functional organization across subjects.

## References
- Haxby, J. V., Guntupalli, J. S., Connolly, A. C., Halchenko, Y. O., Conroy, B. R., Gobbini, M. I., Hanke, M., & Ramadge, P. J. (2011). [A common, high-dimensional model of the representational space in human ventral temporal cortex](https://doi.org/10.1016/j.neuron.2011.08.026). *Neuron*, *72*(2), 404–416. 

- Guntupalli, J. S., Hanke, M., Halchenko, Y. O., Connolly, A. C., Ramadge, P. J., & Haxby, J. V. (2016). [A model of representational spaces in human cortex](https://doi.org/10.1093/cercor/bhw068). *Cerebral Cortex*, *26*(6), 2919–2934.

## Repository Structure

```
├── analysis/                               # Perform cross hyperalignment and inspect the hyperalignment results 

├── hyperalignment/                         # Core implementation scripts for hyperalignment
    ├── HyperAlign_utils.py                 # Support functions for preprocessing 
    ├── HyperAlign_class.py                 # HyperAlign class and SL_Hyperalign class for performing (searchlight)-hyperalignment 

├── main/                                   # Perform hyperalignment 
    ├── main_CHA_train.py                   # Perform connectivity-based hyperalignment
    ├── main_CHA_train.sh                   # Perform connectivity-based hyperalignment (for submisson on clusters)
    ├── main_prep-test-data_CHA.py          # Preprocessing test data without performing hyperalignment
    ├── main_prep-test-data_CHA.sh          # Preprocessing test data without performing hyperalignment (for submisson on clusters)
    ├── main_prep-test-data_RHA.py          # Preprocessing test data without performing hyperalignment
    ├── main_prep-test-data_RHA.sh          # Preprocessing test data without performing hyperalignment (for submisson on clusters)
    ├── main_RHA_train.py                   # Perform response-based hyperalignment
    ├── main_RHA_train.sh                   # Perform response-based hyperalignment (for submisson on clusters)

└── README.md                               # This file 

```
Note: Examples usages in main/ are based on NSD. 