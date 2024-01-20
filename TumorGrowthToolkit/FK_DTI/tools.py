#%%
from os.path import expanduser, join

import matplotlib.pyplot as plt
import nibabel as nib

from dipy.core.gradients import gradient_table
from dipy.data import fetch_sherbrooke_3shell
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti

from dipy.reconst.dti import TensorModel
import nibabel as nib
import os
import scipy.ndimage as ndi
from dipy.reconst.dti import fractional_anisotropy, color_fa

import numpy as np


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

if __name__ == "__main__":
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
    plt.imshow(rgb[:,:,25])
# %%
