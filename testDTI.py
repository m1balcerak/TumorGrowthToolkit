#%%
from TumorGrowthToolkit.FK_DTI import FK_DTI_Solver
from TumorGrowthToolkit.FK import Solver as FK_Solver
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import time
import scipy.ndimage
import nibabel as nib
import os
import torch
from scipy.ndimage import binary_dilation


def elongate_tensor_along_main_axis(D, scale_factor):
    """
    Elongates a diffusion tensor along its main axis.

    Parameters:
    - D: A 3x3 diffusion tensor (numpy array).
    - scale_factor: The factor by which to scale the largest eigenvalue.

    Returns:
    - D_prime: The new diffusion tensor elongated along its main axis.
    """
    # Perform eigen decomposition of the diffusion tensor
    eigenvalues, eigenvectors = np.linalg.eigh(D)
    
    # Find the index of the largest eigenvalue
    max_eigenvalue_index = np.argmax(eigenvalues)
    
    # Scale the largest eigenvalue
    eigenvalues[max_eigenvalue_index] *= scale_factor
    
    # Reconstruct the diffusion tensor
    D_prime = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    return D_prime


def elongate_tensor_along_main_axis_torch(tensor_array, scale_factor):
    # Ensure input is a float tensor for eigen decomposition
    tensor_array = tensor_array.float()
    
    # Compute the eigenvalues and eigenvectors for each 3x3 matrix
    e, v = torch.linalg.eigh(tensor_array)
    
    # Identify the maximum eigenvalue for each matrix
    max_eigenvalue_indices = torch.argmax(e, dim=-1, keepdim=True)
    max_eigenvalues = torch.gather(e, -1, max_eigenvalue_indices)
    
    # Scale the maximum eigenvalues by the scale factor
    scaled_max_eigenvalues = max_eigenvalues * scale_factor
    
    # Update the eigenvalues tensor with the scaled values
    e_updated = e.scatter_(-1, max_eigenvalue_indices, scaled_max_eigenvalues)
    
    # Reconstruct the tensors from the eigenvectors and the scaled eigenvalues
    tensor_array_prime = v @ torch.diag_embed(e_updated) @ v.transpose(-2, -1)
    
    return tensor_array_prime
import torch

def elongate_tensor_along_main_axis_torch_adjusted(tensor_array, scale_factor):
    tensor_array = tensor_array.float()
    e, v = torch.linalg.eigh(tensor_array)

    # Original sum of eigenvalues
    original_sum = torch.sum(e, dim=-1, keepdim=True)

    # Identify and scale the maximum eigenvalue
    max_eigenvalue_indices = torch.argmax(e, dim=-1, keepdim=True)
    max_eigenvalues = torch.gather(e, -1, max_eigenvalue_indices)
    scaled_max_eigenvalues = max_eigenvalues * scale_factor
    
    # Calculate the difference introduced by scaling
    difference = scaled_max_eigenvalues - max_eigenvalues

    # Prepare to adjust the other eigenvalues to keep the sum constant
    adjustment = difference / 2
    mask = torch.ones_like(e, dtype=torch.bool)
    mask.scatter_(-1, max_eigenvalue_indices, 0)  # Mask out the max eigenvalue

    # Adjust the other two eigenvalues
    e_adjusted = torch.where(mask, e - adjustment, e)
    e_adjusted_sum = torch.sum(e_adjusted, dim=-1, keepdim=True)
    
    # Calculate final adjustments due to precision errors
    final_adjustment = (original_sum - e_adjusted_sum) / 3
    e_final = e_adjusted + torch.where(mask, final_adjustment, torch.zeros_like(final_adjustment))

    # Ensure the scaled max eigenvalue is set correctly
    e_final.scatter_(-1, max_eigenvalue_indices, scaled_max_eigenvalues)

    # Reconstruct the tensor
    tensor_array_prime = v @ torch.diag_embed(e_final) @ v.transpose(-2, -1)

    return tensor_array_prime

#%%
# Apply a Gaussian filter for smooth transitions
tissue = nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/rgbResults/sub-tgm051_ses-preop_space-sri_dti_RGB.nii.gz").get_fdata()
print('shape: (x, y, z, fa-diffusion) :', tissue.shape)

seg = nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/tgm/tgm051/preop/sub-tgm051_ses-preop_space-sri_seg.nii.gz").get_fdata()

brainTissue = nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/tgm/tgm051/preop/sub-tgm051_ses-preop_space-sri_tissuemask.nii.gz").get_fdata()

tissueTensor= nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/rgbResults/sub-tgm051_ses-preop_space-sri_dti_tensor.nii.gz").get_fdata()

brainMask = nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/tgm/tgm051/preop/sub-tgm051_ses-preop_space-sri_brainmask.nii.gz").get_fdata()

#normalize the tensor
tissueTensor = tissueTensor/np.max(tissueTensor.flatten())

affine = nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/tgm/tgm051/preop/sub-tgm051_ses-preop_space-sri_seg.nii.gz").affine
# only use diagonal elements.

CSFMask = binary_dilation(brainTissue == 1, iterations = 1)

tissue[CSFMask] = 0
tissueTensor[CSFMask] = 0
#%%
def makeXYZ_rgb_from_tensor(tensor, brainMask):
    
    output = np.zeros(tissue.shape)
    output[:,:,:,0] = tensor[:,:,:,0,0]
    output[:,:,:,1] = tensor[:,:,:,1,1]
    output[:,:,:,2] = tensor[:,:,:,2,2]

    #set the mean to 0.2 and clip at 1 for stability reasons
    output /= np.mean(output[brainMask >0])#.flatten()[output.flatten()>0.0])
    output *= 0.2
    output[output>1] = 1
    output[output<0] = 0

    return output

tissueFromTensor = makeXYZ_rgb_from_tensor(tissueTensor, brainMask)
#%%
# Scale factor to elongate the tensor along its main axis
scale_factor = 2#1.5 # 250000.0

# Apply the transformation
tensor_array_prime = elongate_tensor_along_main_axis_torch_adjusted(torch.from_numpy(tissueTensor), scale_factor).numpy()

tissueFromScaledTensor = makeXYZ_rgb_from_tensor(tensor_array_prime, brainMask)
#%%
plt.title('Tensor tissue')
plt.imshow((tissueFromTensor/np.max(tissueFromTensor))[:,:,75,:] )
plt.show()
plt.title('Scaled Tensor tissue by factor: ' + str(scale_factor) )
plt.imshow((tissueFromScaledTensor/np.max(tissueFromScaledTensor))[:,:,75,:] )
plt.show()
plt.title('')
plt.hist(tissueFromTensor[brainMask>0].flatten(), bins=100, alpha=0.5, label='tissueFromTensor')
plt.hist(tissueFromScaledTensor[brainMask>0].flatten(), bins=100, alpha=0.5, label='tissueFromScaledTensor_' + str(scale_factor))
plt.legend(loc='upper right')
print("sum of tissueFromTensor: ", np.sum(tissueFromTensor[brainMask>0].flatten()))
print("sum of tissueFromScaledTensor: ", np.sum(tissueFromScaledTensor[brainMask>0].flatten()))
# %%
parameters = {
    'Dw': 0.7,          # maximum diffusion coefficient
    'rho': 0.4,        # Proliferation rate
    'rgb':tissueFromScaledTensor, # diffusion tissue map as shown above
    'diffusionTensorExponent': 1.0, # exponent for the diffusion tensor, 1.0 for linear relationship
    'NxT1_pct': 0.45,    # tumor position [%]
    'NyT1_pct': 0.32,
    'NzT1_pct': 0.60,
    'init_scale': 1., #scale of the initial gaussian
    'resolution_factor': 1, #resultion scaling for calculations
    'verbose': True, #printing timesteps 
    'time_series_solution_Nt': 64 # number of timesteps in the output
}

x = int(tissue.shape[0]*parameters["NxT1_pct"])
y = int(tissue.shape[1]*parameters["NyT1_pct"])
z = int(tissue.shape[2]*parameters["NzT1_pct"])

com = scipy.ndimage.measurements.center_of_mass(seg)

plt.imshow(brainTissue[:,:,z])
plt.show()

plt.imshow(tissue[:,:,z])
#plt.imshow(seg[:,:,z],alpha=0.5)
plt.title('Fractional Anisotropy Tissue')
plt.xlabel('y')
plt.ylabel('x')
plt.scatter(y,x, c='r')
plt.show()
#plt.imshow(seg[:,:,z],alpha=0.5)
#%%
# Run the FK_solver and plot the results
start_time = time.time()
fK_DTI_Solver = FK_DTI_Solver(parameters)
result = fK_DTI_Solver.solve()
end_time = time.time()  # Store the end time
execution_time = int(end_time - start_time)  # Calculate the difference

print(f"Execution Time: {execution_time} seconds")
if result['success']:
    print("Simulation successful!")
else:
    print("Error occurred:", result['error'])
# Create custom color maps
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['black', 'white'], 256)
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap2', ['black', 'green', 'yellow', 'red'], 256)


#%% run normal FK
gm = brainTissue == 2
wm = brainTissue == 3
gm[CSFMask] = 0
wm[CSFMask] = 0
parametersFK = {
    'Dw': 0.7,          # maximum diffusion coefficient
    'rho': 0.3,        # Proliferation rate
    'gm' : gm,
    'wm' : wm,
    'NxT1_pct': 0.45,    # tumor position [%]
    'NyT1_pct': 0.32,
    'NzT1_pct': 0.60,
    'init_scale': 1., #scale of the initial gaussian
    'resolution_factor':1, #resultion scaling for calculations
    'verbose': True, #printing timesteps 
    'time_series_solution_Nt': 64 # number of timesteps in the output
}
fkSolver = FK_Solver(parametersFK)
resultFK = fkSolver.solve()
#%%
# Calculate the slice index
NzT = int(parameters['NzT1_pct'] * tissue.shape[2]) +5

def plot_time_series(wm_data, time_series_data, slice_index):
    plt.figure(figsize=(24, 12))

    # Generate 8 indices evenly spaced across the time series length
    time_points = np.linspace(0, time_series_data.shape[0] - 1, 8, dtype=int)

    for i, t in enumerate(time_points):
        plt.subplot(2, 4, i + 1)  # 2 rows, 4 columns, current subplot index
        plt.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
        plt.imshow(time_series_data[t, :, :, slice_index], cmap=cmap2, vmin=0, vmax=1, alpha=0.65)
        plt.title(f"Time Slice {t + 1}")

    plt.tight_layout()
    plt.show()

plot_time_series(np.mean(tissue, axis=3),result['time_series'], NzT)
plot_time_series(np.mean(tissue, axis=3),resultFK['time_series'], NzT)
#%% 
plt.imshow(brainTissue[:,:,z]>0,alpha=0.5*(brainTissue[:,:,z]==0), cmap='gray')
plt.imshow(seg[:,:,z],alpha=0.5*(seg[:,:,z]>0), cmap='Greens')  
plt.imshow(result['final_state'][:,:,z], alpha=0.5*(result['final_state'][:,:,z]>0.001), cmap = "Reds")	

plt.title('Tumor')

#%% save results
path = "/mnt/8tb_slot8/jonas/workingDirDatasets/tgm/dtiFirstTests/tgm051/"
os.makedirs(path, exist_ok=True)
nib.save(nib.Nifti1Image(result['final_state'], affine=affine), path + "resultTensor.nii.gz")
nib.save(nib.Nifti1Image(tissueFromTensor, affine=affine), path + "tissueFromTensor.nii.gz")
# %%