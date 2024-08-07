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

#%%
#tissueSegmentationPath ="dataset/sub-mni152_tissue-with-antsN4_space-sri.nii.gz"
#tensorPath = "dataset/FSL_HCP1065_tensor_1mm_space-HPC-AntsIndexSpace_SRI.nii.gz"

tissueSegmentationPath = "../../dataset/sub-mni152_tissue-with-ants_space-sri.nii.gz"
tensorPath = "../../dataset/FSL_HCP1065_tensor_1mm_space-HPC-AntsIndexSpace_SRI.nii.gz"


originalTissue = nib.load(tissueSegmentationPath).get_fdata()
affine = nib.load(tissueSegmentationPath).affine

# load tensor
# this is a 3x3 symmetric tensor
tissueTensorRegistered = nib.load(tensorPath).get_fdata()[:,:,:,0,:]
#%%
for i in range(6):
    plt.hist(tissueTensorRegistered[:,:,:,i].flatten(), bins=100)
    print(np.mean(tissueTensorRegistered[:,:,:,i].flatten()), np.std(tissueTensorRegistered[:,:,:,i].flatten()))
    plt.title(f"histogram of tensor element {i}")
    plt.show()

tissueTensor = tools.get_tensor_from_lower6(tissueTensorRegistered)

#%%
CSFMask = originalTissue == 1 # binary_dilation(originalTissue == 1, iterations = 1)

tissue = originalTissue.copy()
tissue[CSFMask] = 0
tissueTensor[CSFMask] = 0

plt.imshow(tissue[:,:,75])
plt.show()
plt.imshow(CSFMask[:,:,75] * 0.5)

# %%
dw = 1
rho = 0.2
#ventricle
"""x = 0.6
y = 0.3
z = 0.50"""

x = 0.6
y = 0.6
z = 0.55

init_scale = 0.1
resolution_factor = 0.5# 0.6#1
stoppingVolume =  20000 #10000000#100000
parameters = {
    'Dw': dw,          # maximum diffusion coefficient
    'rho': rho,        # Proliferation rate
    'diffusionTensors':tissueTensor, # diffusion tissue map as shown above
    'diffusionTensorExponent': 1, # exponent for the diffusion tensor, 1.0 for linear relationship
    'diffusionEllipsoidScaling':1,#21.713178343886213,
    'NxT1_pct': x,    # tumor position [%]
    'NyT1_pct': y,
    'NzT1_pct': z,
    'init_scale': init_scale, #scale of the initial gaussian
    'resolution_factor': resolution_factor, #resultion scaling for calculations
    'verbose': True, #printing timesteps 
    'time_series_solution_Nt': 64, # number of timesteps in the output
    'stopping_volume': stoppingVolume,
    'stopping_time': 1000,
}

x = int(tissue.shape[0]*parameters["NxT1_pct"])
y = int(tissue.shape[1]*parameters["NyT1_pct"])
z = int(tissue.shape[2]*parameters["NzT1_pct"])

com = 100#scipy.ndimage.measurements.center_of_mass(seg)

plt.imshow(tissue[:,:,z])
plt.show()


#plt.imshow((rgbFromTensor[:,:,z] / np.max(rgbFromTensor[:,:,z])))

plt.imshow(tissueTensor[:,:,z,2,2])

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
result = fK_DTI_Solver.solve(doPlot=True)
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
#%%
# Calculate the slice index
NzT = int(parameters['NzT1_pct'] * tissue.shape[2]) 

def plot_time_series(wm_data, time_series_data, slice_index = None):
    if slice_index is None:
        # com
        slice_index = int(scipy.ndimage.measurements.center_of_mass(time_series_data[-1])[2])
    plt.figure(figsize=(24, 12))
    print("Final Volume = " + str(np.sum(time_series_data[-1]/1000)))

    # Generate 8 indices evenly spaced across the time series length
    time_points = np.linspace(0, time_series_data.shape[0] - 1, 8, dtype=int)

    for i, t in enumerate(time_points):
        plt.subplot(2, 4, i + 1)  # 2 rows, 4 columns, current subplot index
        plt.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
        plt.imshow(time_series_data[t, :, :, slice_index], cmap=cmap2, vmin=0, vmax=1, alpha=0.65)
        plt.title(f"Time Slice {t + 1}")

    plt.tight_layout()
    plt.show()

gm = tissue == 2
wm = tissue == 3
plot_time_series(wm+0.5*gm,result['time_series'])
#%% run normal FK
gm = tissue == 2
wm = tissue == 3
gm[CSFMask] = 0
wm[CSFMask] = 0
parametersFK = {
    'Dw': dw,          # maximum diffusion coefficient
    'rho': rho,        # Proliferation rate
    'gm' : gm,
    'wm' : wm,
    'NxT1_pct': parameters["NxT1_pct"],    # tumor position [%]
    'NyT1_pct': parameters["NyT1_pct"],
    'NzT1_pct': parameters["NzT1_pct"],
    'init_scale': init_scale, #scale of the initial gaussian
    'resolution_factor': resolution_factor, #resultion scaling for calculations
    'verbose': True, #printing timesteps 
    'time_series_solution_Nt': 64, # number of timesteps in the output
    'stopping_volume': stoppingVolume,
    'stopping_time': 1000,
}

#%%
fkSolver = FK_Solver(parametersFK)
resultFK = fkSolver.solve()
#%%
plot_time_series(wm+gm*0.5,result['time_series'], z)
plot_time_series(wm+gm*0.5,resultFK['time_series'],z)
#%% 
plt.imshow(tissue[:,:,z]>0,alpha=0.5*(tissue[:,:,z]==0), cmap='gray')
#plt.imshow(seg[:,:,z],alpha=0.5*(seg[:,:,z]>0), cmap='Greens')  
plt.imshow(result['final_state'][:,:,z], alpha=0.5*(result['final_state'][:,:,z]>0.001), cmap = "Reds")	
plt.colorbar()
plt.title('Tumor DTI')
plt.show()
plt.imshow(tissue[:,:,z]>0,alpha=0.5*(tissue[:,:,z]==0), cmap='gray')
#plt.imshow(seg[:,:,z],alpha=0.5*(seg[:,:,z]>0), cmap='Greens')  
plt.imshow(resultFK['final_state'][:,:,z], alpha=0.5*(resultFK['final_state'][:,:,z]>0.001), cmap = "Reds")	

plt.title('Tumor FK')
plt.colorbar()
plt.show()
#%%

path = "./"
os.makedirs(path, exist_ok=True)
nib.save(nib.Nifti1Image(result['final_state'], affine=affine), path + "resultTensor_Strength" + str(round(parameters['diffusionEllipsoidScaling'],2)).replace(".", "_") +"_x_" + str(round(parameters['NxT1_pct'] , 2) ) + "_stoppingVolume_" + str(stoppingVolume) + ".nii.gz")
#save fk
nib.save(nib.Nifti1Image(resultFK['final_state'], affine=affine), path + "resultFK"+"_x_" + str(round(parameters['NxT1_pct'] , 2) ) + "_stoppingVolume_" + str(stoppingVolume) + ".nii.gz")
#%%
print("FK volume = " + str(np.sum(resultFK['final_state'])))
print("DTI volume = " + str(np.sum(result['final_state'])))
# %%
