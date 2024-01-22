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

#%%
tissue = nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/rgbResults/sub-tgm051_ses-preop_space-sri_dti_RGB.nii.gz").get_fdata()
print('shape: (x, y, z, fa-diffusion) :', tissue.shape)

seg = nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/tgm/tgm051/preop/sub-tgm051_ses-preop_space-sri_seg.nii.gz").get_fdata()

brainTissue = nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/tgm/tgm051/preop/sub-tgm051_ses-preop_space-sri_tissuemask.nii.gz").get_fdata()

tissueTensor= nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/rgbResults/sub-tgm051_ses-preop_space-sri_dti_tensor.nii.gz").get_fdata()

brainMask = nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/tgm/tgm051/preop/sub-tgm051_ses-preop_space-sri_brainmask.nii.gz").get_fdata()

#normalize the tensor
#tissueTensor = tissueTensor/np.max(tissueTensor.flatten())

affine = nib.load("/mnt/8tb_slot8/jonas/datasets/TGM/tgm/tgm051/preop/sub-tgm051_ses-preop_space-sri_seg.nii.gz").affine
# only use diagonal elements.

CSFMask = binary_dilation(brainTissue == 1, iterations = 1)

tissue[CSFMask] = 0
tissueTensor[CSFMask] = 0
#%%
plt.imshow(brainTissue[:,:,75])
plt.show()
plt.imshow(CSFMask[:,:,75] * 0.5)

# %%
parameters = {
    'Dw': 0.19540936796730088,          # maximum diffusion coefficient
    'rho': 3.432108393857487,        # Proliferation rate
    'diffusionTensors':tissueTensor, # diffusion tissue map as shown above
    'diffusionTensorExponent': 1.0, # exponent for the diffusion tensor, 1.0 for linear relationship
    'diffusionEllipsoidScaling': 21.713178343886213,
    'NxT1_pct': 0.45713315234845947,    # tumor position [%]
    'NyT1_pct': 0.3436560371780158,
    'NzT1_pct': 0.595781739465849,
    'init_scale': 1., #scale of the initial gaussian
    'resolution_factor': 1.0, #resultion scaling for calculations
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
# Run the DTI_FK_solver and plot the results
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
    'rho': 0.1,        # Proliferation rate
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
NzT = int(parameters['NzT1_pct'] * tissue.shape[2]) 

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