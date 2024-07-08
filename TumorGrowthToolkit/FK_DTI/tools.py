#%%
from os.path import expanduser, join

import matplotlib.pyplot as plt
import nibabel as nib

from dipy.core.gradients import gradient_table
from dipy.data import fetch_sherbrooke_3shell, fetch_bundle_atlas_hcp842
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti

from dipy.reconst.dti import TensorModel
import nibabel as nib
import os
import scipy.ndimage as ndi
from dipy.reconst.dti import fractional_anisotropy, color_fa

import numpy as np

import gc

import numpy as np
import numpy as np
from scipy.linalg import eigh, diagsvd

def get_RGB(dataFolder, dtiPath = 'DTI.nii.gz', bvalsPath = 'DTI.bval', bvecsPath= 'DTI.bvec', brainmaskPath = None, saveFolder= None,doSave = True, maskthreshold = 20000):
    print('loading data')

    if saveFolder == None:
        saveFolder = dataFolder
    try:
        nifti = nib.load(os.path.join(dataFolder,dtiPath))
    
    except:
        print('failed to load dti nifti')
        return False
    niftiimg = nifti.get_fdata()
    affine = nifti.affine

    bvals, bvecs = read_bvals_bvecs(os.path.join(dataFolder,bvalsPath), os.path.join(dataFolder,bvecsPath))

    gtabnifti = gradient_table(bvals, bvecs)

    tenmodel = TensorModel(gtabnifti)
    maskedNifti = np.zeros(niftiimg.shape)

    if brainmaskPath == None:
        from dipy.segment.mask import median_otsu

        # Apply median_otsu to generate the mask. Adjust 'median_radius' and 'num_pass' as needed.
        _, mask = median_otsu(np.mean(niftiimg, axis= -1), median_radius=4, numpass=4)

        mask  = np.mean(niftiimg, axis = -1)>maskthreshold
    else: 
        mask = np.array(nib.load(os.path.join(dataFolder,brainmaskPath)).get_fdata()) == 1

    plt.imshow(mask[:,:,40])
    plt.show()
    
    maskedNifti[mask] = niftiimg[mask]

    plt.imshow(np.mean(maskedNifti, axis=-1)[:,:,40])
    plt.colorbar()
    plt.show()

    #start fit
    tenfit = tenmodel.fit(maskedNifti)
    #fitDone

    FA = np.clip(tenfit.fa, 0, 1) 
    #todo understand color_fa
    RGB =  color_fa(FA, tenfit.evecs)

    if doSave:

        rgb_img = nib.Nifti1Image(RGB, affine)
        savePath = dtiPath.split('.')[0] + '_RGB.nii.gz'	

        nib.save(rgb_img, os.path.join(saveFolder,savePath))

        tesnor_img = nib.Nifti1Image(tenfit.quadratic_form, affine)
        savePath = dtiPath.split('.')[0] + '_tensor.nii.gz'	

        nib.save(tesnor_img, os.path.join(saveFolder,savePath))
        print('saved')

        return True

    return RGB

def get_tensor_from_lower6(lower6):
    #[dxx, dxy, dyy, dxz, dyz, dzz]

    tensor  = np.zeros(lower6.shape[0:3] + (3,3))#.astype(np.string_) for testing
    print(tensor.shape)
    tensor[..., 0, 0] = lower6[..., 0]
    tensor[..., 1, 1] = lower6[..., 2]
    tensor[..., 2, 2] = lower6[..., 5]
    tensor[..., 0, 1] = lower6[..., 1]
    tensor[..., 1, 0] = lower6[..., 1]
    tensor[..., 0, 2] = lower6[..., 3]
    tensor[..., 2, 0] = lower6[..., 3]
    tensor[..., 1, 2] = lower6[..., 4]
    tensor[..., 2, 1] = lower6[..., 4]

    return tensor


def elongate_tensor_along_main_axis_torch(tensor_arrayNP, scale_factor):
    import  torch
    torch.no_grad()
    tensor_array = torch.from_numpy(tensor_arrayNP)

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

    e_final_detached = e_final.detach()
    v_detached = v.detach()
    tensor_array_prime = tensor_array_prime.detach()
    numpyRet = tensor_array_prime.numpy()

    del tensor_array
    del e
    del v
    del e_adjusted
    del e_adjusted_sum
    del e_final
    del e_final_detached
    del v_detached
    del tensor_array_prime
 
    gc.collect()

    return numpyRet

def makeXYZ_rgb_from_tensor(tensor):
    
    output = np.zeros(tensor.shape[:4])
    output[:,:,:,0] = tensor[:,:,:,0,0]
    output[:,:,:,1] = tensor[:,:,:,1,1]
    output[:,:,:,2] = tensor[:,:,:,2,2]

    brainMask = np.max(output, axis=-1) > 0

    #set the mean to 1 and clip at 10 for stability reasons
    output /= np.mean(output[brainMask >0])#.flatten()[output.flatten()>0.0])
    output *= 1#0.2
    output[output>10] = 10
    output[output<0] = 0

    return output

if False:# __name__ == "__main__":
    # %% generate_example_DTI_Image
    ###############################################################################
    # With the following commands we can download a dMRI dataset
    fetch_sherbrooke_3shell()

    #%%

    home = expanduser('~')
    dname = join(home, '.dipy', 'sherbrooke_3shell')

    fdwi = join(dname, 'HARDI193.nii.gz')

    print(fdwi)

    fbval = join(dname, 'HARDI193.bval')

    print(fbval)

    fbvec = join(dname, 'HARDI193.bvec')

    print(fbvec)
    # %%
    rgb = get_RGB(dname, dtiPath = 'HARDI193.nii.gz', bvalsPath = 'HARDI193.bval', bvecsPath= 'HARDI193.bvec', brainmaskPath = None, saveFolder= "./dataset", doSave = False, maskthreshold=50)
    # %%

    
    atlas = fetch_bundle_atlas_hcp842()
    tensor_eigenvalues, affine = load_nifti(atlas['fa'])

    plt.imshow(rgb[:,:,25])
# %%
