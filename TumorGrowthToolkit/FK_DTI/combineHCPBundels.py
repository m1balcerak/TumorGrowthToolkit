#%%
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt



# %%
folder = "/mnt/8tb_slot8/jonas/datasets/HCPprob/prob/"
# load all nifty files of folder into a list of numpy arrays
niftyList = []
files = os.listdir(folder)
for file in files:
    if file.endswith(".nii.gz"):
        niftyList.append(nib.load(os.path.join(folder,file)).get_fdata())

# %%
affine = nib.load(os.path.join(folder,files[0])).affine
# %%
sum =  np.sum(niftyList, axis=0)

# %%
plt.imshow(sum[:,:,40])
#save sum as nifti
resultFolder = "/mnt/8tb_slot8/jonas/workingDirDatasets/HCPprob/prob/"
os.makedirs(resultFolder, exist_ok=True)
nib.save(nib.Nifti1Image(sum, affine), os.path.join(resultFolder,"sum.nii.gz"))

# %%
