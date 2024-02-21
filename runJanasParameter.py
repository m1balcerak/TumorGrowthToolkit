#%%
from TumorGrowthToolkit.FK import Solver as FKSolver
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import time
import scipy.ndimage
import nibabel as nib

# Apply a Gaussian filter for smooth transitions - those are the data from michal. Janas are oriented differently...
wm_data = nib.load("/mnt/8tb_slot8/jonas/datasets/Jana_IEEETMI_n8/P3/preop/sub-P3_ses-preop_space-jana_seg-wm.nii.gz").get_fdata()
gm_data = nib.load("/mnt/8tb_slot8/jonas/datasets/Jana_IEEETMI_n8/P3/preop/sub-P3_ses-preop_space-jana_seg-gm.nii.gz").get_fdata()

segmt1 = nib.load("/mnt/8tb_slot8/jonas/datasets/Jana_IEEETMI_n8/P3/preop/sub-P3_ses-preop_space-jana_seg-t1c.nii.gz").get_fdata()

segmFlair = nib.load("/mnt/8tb_slot8/jonas/datasets/Jana_IEEETMI_n8/P3/preop/sub-P3_ses-preop_space-jana_seg-flair.nii.gz").get_fdata()

map  = nib.load("/mnt/8tb_slot8/jonas/datasets/Jana_IEEETMI_n8/P3/preop/sub-P3_ses-preop_space-jana_tgm-map.nii.gz").get_fdata()

affine = nib.load("/mnt/8tb_slot8/jonas/datasets/Jana_IEEETMI_n8/P3/preop/sub-P3_ses-preop_space-jana_seg-wm.nii.gz").affine

#%%
gm_data.shape

#%%
wm_data.shape

#%%
janasPaperX, janasPaperY, janasPaperZ =   169.293, 177.075, 141.235
Janas_x, Janas_y, Janas_z =  janasPaperZ, janasPaperX, janasPaperY
janasStartPixel = np.zeros_like(wm_data)
janasStartPixel[int(Janas_x),:, :] = 1
janasStartPixel[:,int(Janas_y), :] = 1
janasStartPixel[:,:,int(Janas_z)] = 1
#plt.imshow(gm_data[:,:,100], cmap='gray')
plt.imshow(segmFlair[:,:,140], alpha = 0.5)
plt.imshow(janasStartPixel[:,:,140], alpha = 0.5, cmap='Reds')

#plt.scatter(y,x)
#plt.imshow(map[:,:,160], alpha = 0.5)
#plt.imshow(wm_data[:,:,75])

#center of mass
#%%

comorg = scipy.ndimage.measurements.center_of_mass(segmFlair) 
com = comorg / np.array(segmt1.shape)
print(com)


janasStartPixel = np.zeros_like(wm_data)
janasStartPixel[int(comorg[0]),:, :] = 1
janasStartPixel[:,int(comorg[1]), :] = 1
janasStartPixel[:,:,int(comorg[2])] = 1
#plt.imshow(gm_data[:,:,100], cmap='gray')
plt.imshow(segmFlair[:,:,140], alpha = 0.5)
plt.imshow(janasStartPixel[:,:,140], alpha = 0.5, cmap='Reds')

#%%
# Set up parameters
janasRoh = 0.029
janasDw = 0.188
janasT = 273.13

Tsolver = 100

solverDw = janasDw * janasT / Tsolver
solverRoh = janasRoh *  janasT / Tsolver

parameters = {
    'Dw': solverDw,     # Diffusion coefficient for white matter
    'rho': solverRoh,   # Proliferation rate
    'RatioDw_Dg': 10,   # Ratio of diffusion coefficients in white and grey matter
    'gm': gm_data,      # Grey matter data
    'wm': wm_data,      # White matter data
    'NxT1_pct': com[0],    # tumor position [%]
    'NyT1_pct': com[1],
    'NzT1_pct': com[2],
    'init_scale': 1., #scale of the initial gaussian
    'resolution_factor': 1, #resultion scaling for calculations
    'th_matter': 0.1, #when to stop diffusing: at th_matter > gm+wm
    'verbose': True, #printing timesteps 
    'time_series_solution_Nt': 64 # number of timesteps in the output
    
}

# Create custom color maps
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['black', 'white'], 256)
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap2', ['black', 'green', 'yellow', 'red'], 256)

# Calculate the slice index
NzT = int(parameters['NzT1_pct'] * gm_data.shape[2])

# Plotting function
def plot_tumor_states(wm_data, initial_state, final_state, slice_index):
    plt.figure(figsize=(12, 6))

    # Plot initial state
    plt.subplot(1, 2, 1)
    plt.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
    plt.imshow(initial_state[:, :, slice_index], cmap=cmap2, vmin=0, vmax=1, alpha=0.65)
    plt.title("Initial Tumor State")

    # Plot final state
    plt.subplot(1, 2, 2)
    plt.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
    plt.imshow(final_state[:, :, slice_index], cmap=cmap2, vmin=0, vmax=1, alpha=0.65)
    plt.title("Final Tumor State")
    plt.show()
    

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
# %%
# Run the FK_solver and plot the results
start_time = time.time()
fk_solver = FKSolver(parameters)
result = fk_solver.solve()
end_time = time.time()  # Store the end time
execution_time = int(end_time - start_time)  # Calculate the difference

print(f"Execution Time: {execution_time} seconds")
if result['success']:
    print("Simulation successful!")
    plot_tumor_states(wm_data, result['initial_state'], result['final_state'], NzT)
    plot_time_series(wm_data,result['time_series'], NzT)
else:
    print("Error occurred:", result['error'])

#%%
plt.imshow((map- result["final_state"]) [:,:,int(comorg[2] - 20)], alpha = 0.5, cmap='bwr', vmin=-1, vmax=1)

#%%
plt.imshow(result["final_state"] [:,:,int(comorg[2])], alpha = 0.5, cmap='Blues')
#%%
#plt.imshow(map [:,:,int(comorg[2])], alpha = 0.5, cmap='Reds')
plt.imshow(np.mean(result["final_state"], axis=-1 ), alpha = 0.5, cmap='Blues')

plt.imshow(janasStartPixel[:,:,int(comorg[2])], alpha = 0.5, cmap='Reds')

# %%
#saveNifti
nib.save(nib.Nifti1Image(result['final_state'], affine), '/mnt/8tb_slot8/jonas/datasets/Jana_IEEETMI_n8/P3/preop/ourSolutionWithMap.nii.gz')
# %%
# %%
