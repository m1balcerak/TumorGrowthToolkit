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
from scipy.ndimage import binary_dilation
import TumorGrowthToolkit.FK_DTI.tools as tools


from matplotlib.colors import LinearSegmentedColormap
colors = ["#FBB760", "#F00F0F"]  # RGB values for orange and red
n_bins = 100  # Number of bins for the color map
cmap_name = 'orange_red'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# %%
tissueSegmentationPath = "./dataset/sub-mni152_tissue-with-antsN4_space-sri.nii.gz"
tensorPath = "./dataset/FSL_HCP1065_tensor_1mm_space-HPC-AntsIndexSpace_SRI.nii.gz"

originalTissue = nib.load(tissueSegmentationPath).get_fdata()
affine = nib.load(tissueSegmentationPath).affine

#create a 3x3 tensor for each voxel
tissueTensor = tools.get_tensor_from_lower6(nib.load(tensorPath).get_fdata()[:,:,:,0,:])


CSFMask = originalTissue == 1 # binary_dilation(originalTissue == 1, iterations = 1)

tissue = originalTissue.copy()
tissue[CSFMask] = 0
tissueTensor[CSFMask] = 0

plt.imshow(tissue[:,:,75])
plt.title("Tissue segmentation")

gm = tissue == 2
wm = tissue == 3
gm[CSFMask] = 0
wm[CSFMask] = 0
gm, wm = gm * 1.0, wm * 1.0

tissue[wm + gm <= 0] = 0

# %%

dw = 3
rho = 0.6#2
#ventricle
"""x = 0.6
y = 0.3
z = 0.50"""
Nx = 0.45
Ny = 0.35
Nz = 0.55

init_scale = 0.1#0.1
resolution_factor = 0.5# 0.6#1
stoppingVolume =  90500
stoppingTime = 1100000
ratioDW_Dg = 10
parameters = {
    'Dw': dw,          # maximum diffusion coefficient
    'rho': rho,        # Proliferation rate
    'diffusionTensors':tissueTensor, # diffusion tissue map as shown above
    'diffusionTensorExponent': 1, # exponent for the diffusion tensor, 1.0 for linear relationship
    'diffusionEllipsoidScaling':1,#21.713178343886213,
    'NxT1_pct': Nx,    # tumor position [%]
    'NyT1_pct': Ny,
    'NzT1_pct': Nz,
    'init_scale': init_scale, #scale of the initial gaussian
    'resolution_factor': resolution_factor, #resultion scaling for calculations
    'verbose': False, #printing timesteps 
    'time_series_solution_Nt': 64, # number of timesteps in the output
    'stopping_volume': stoppingVolume,
    'stopping_time': stoppingTime,
    "use_homogen_gm": True,
    'gm' : gm * 1.0,
    'wm' : wm * 1.0,
    "RatioDw_Dg" : ratioDW_Dg,
    "desiredSTD" : 1,#1#0.5 #0.5#1
    "relative_upper_limit_DTI":2# #TODO
}

x = int(tissue.shape[0]*parameters["NxT1_pct"])
y = int(tissue.shape[1]*parameters["NyT1_pct"])
z = int(tissue.shape[2]*parameters["NzT1_pct"])

#plt.imshow(tissue[:,:,z])
plt.imshow((gm * 0.1 + wm)[:,:,z],  cmap='gray')
plt.scatter(y,x, c='r')
plt.title("Tumor Origin")
plt.show()
# %%
# Run the DTI_FK_solver and plot the results
start_time = time.time()
fK_DTI_Solver = FK_DTI_Solver(parameters)
result = fK_DTI_Solver.solve(doPlot=True)
end_time = time.time()  # Store the end time
execution_time = int(end_time - start_time)  # Calculate the difference

print(f"Execution Time: {execution_time} seconds")
if result['success']:
    print("Simulation successful!")
else:
    print("Error occurred:", result['error'])

plt.imshow(tissue[:,:,z], cmap='gray')
plt.imshow(result['final_state'][:,:,z], alpha=0.5*(result['final_state'][:,:,z]>0.001), cmap = cmap, vmin=0, vmax=1)	
plt.colorbar()
plt.title('Tumor DTI')
plt.show()
# %%

parametersFK = {
    'Dw': dw,          # maximum diffusion coefficient
    'rho': rho,        # Proliferation rate
    'diffusionTensors':tissueTensor, # diffusion tissue map as shown above
    'diffusionTensorExponent': 1, # exponent for the diffusion tensor, 1.0 for linear relationship
    'diffusionEllipsoidScaling':1,#21.713178343886213,
    'NxT1_pct': Nx,    # tumor position [%]
    'NyT1_pct': Ny,
    'NzT1_pct': Nz,
    'init_scale': init_scale, #scale of the initial gaussian
    'resolution_factor': resolution_factor, #resultion scaling for calculations
    'verbose': False, #printing timesteps 
    'time_series_solution_Nt': 64, # number of timesteps in the output
    'stopping_volume': stoppingVolume,
    'stopping_time': stoppingTime,
    "use_homogen_gm": True,
    'gm' : gm * 1.0,
    'wm' : wm * 1.0,
    "RatioDw_Dg" : ratioDW_Dg,
    "desiredSTD" : 0 # must  stay 0 for FK!
}


fkSolver = FK_DTI_Solver(parametersFK)
resultFK = fkSolver.solve()

#%%
plt.imshow(tissue[:,:,z], cmap='gray')
plt.imshow(result['final_state'][:,:,z], alpha=0.5*(result['final_state'][:,:,z]>0.0001), cmap = "Reds", vmin=0, vmax=1)	
plt.colorbar()
plt.title('Tumor DTI')
plt.show()
plt.imshow(tissue[:,:,z], cmap='gray')
plt.imshow(resultFK['final_state'][:,:,z], alpha=0.5*(resultFK['final_state'][:,:,z]>0.0001), cmap = "Reds", vmin=0, vmax=1)	
plt.title('Tumor FK')
plt.colorbar()
plt.show()
plt.imshow((result['final_state']- resultFK['final_state'])[:,:,z], cmap='bwr', vmin=-1, vmax=1)
print("sum difference: ", np.sum(np.abs(result['final_state'] - resultFK['final_state'])))
plt.colorbar()

#%% plot y 
plt.imshow(tissue[:,y,:], cmap='gray')
plt.imshow(result['final_state'][:,y,:], alpha=0.5*(result['final_state'][:,y,:]>0.0001), cmap = "Reds", vmin=0, vmax=1)

plt.colorbar()
plt.title('Tumor DTI')
plt.show()
plt.imshow(tissue[:,y,:], cmap='gray')
plt.imshow(resultFK['final_state'][:,y,:], alpha=0.5*(resultFK['final_state'][:,y,:]>0.0001), cmap = "Reds", vmin=0, vmax=1)
plt.title('Tumor FK')
plt.colorbar()
plt.show()
plt.imshow((result['final_state'][:,y,:] - resultFK['final_state'][:,y,:]), cmap='bwr', vmin=-1, vmax=1)
print("sum difference: ", np.sum(np.abs(result['final_state'] - resultFK['final_state'])))
plt.colorbar()

# %% save DTI
nii = nib.Nifti1Image(result['final_state'], affine)
nib.save(nii, "DTI.nii")
# %% save FK
nii = nib.Nifti1Image(resultFK['final_state'], affine)
nib.save(nii, "FK.nii")


# %%
