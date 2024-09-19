#%%
import numpy as np
import copy
from scipy.ndimage import zoom
from ..FK.FK import Solver as FK_Solver
from . import tools
import scipy.ndimage
from scipy.ndimage import binary_dilation
import nibabel as nib
import matplotlib.pyplot as plt

'''
Forward solver DTI 

1.	It was unnecessary to use the full DTI tensors, as we are simulating on a grid. So, I used the diffusion in each direction (x,y,z) within each voxel. This is equivalent to the colored DTI images.
 
This is how you can generate such an RGB image from DTI:  https://dipy.org/documentation/1.0.0./examples_built/reconst_dti/, but I also included one.

2.	Based on those diffusion values in each direction, I changed the Fisher-Kolmogorov diffusion value along this direction ( ‘get_D_from_DTI()’ ). You can think of many ways to do this. The easiest way would be a proportional mapping, but to suppress low DTI values, I used a polynomial mapping.

By Jonas Weidner - 2023 based on Michal Balcerak solver.
'''

class FK_DTI_Solver(FK_Solver):
    def __init__(self, params):
        super().__init__(params)

    def makeXYZ_rgb_from_tensor(self, tensor, exponent = 1 , linear = 0, wm = None, gm = None, ratioDw_Dg = None, desiredSTD = None, upper_limit = 10):
        output = np.zeros(tensor.shape[:4])

        # use diagonal elements
        output[:,:,:,0] = tensor[:,:,:,0,0]
        output[:,:,:,1] = tensor[:,:,:,1,1]
        output[:,:,:,2] = tensor[:,:,:,2,2]

        output[output<0] = 0

        brainMask = np.max(output, axis=-1) > 0

        plt.imshow(output[:,:,output.shape[2]//2] / np.max(output[:,:,output.shape[2]//2]))
        plt.colorbar()
        plt.title("Before")
        plt.show()

        if wm is not None:
            normalizationMask = wm > 0
        else:
            normalizationMask = brainMask

        if desiredSTD is not None:
            output[brainMask] -= np.mean(output[normalizationMask])
            output[brainMask] /= np.std(output[normalizationMask])
            output[brainMask] *= desiredSTD
            output[brainMask] += 1
        else:
            output[brainMask] /= np.mean(output[normalizationMask])

        output[brainMask] = output[brainMask]**exponent +  output[brainMask] * linear
        plt.imshow(output[:,:,output.shape[2]//2])
        plt.colorbar()
        plt.title("Before 2")
        plt.show()
        
        if not (wm is None or gm  is None or ratioDw_Dg is None):

            print('set gm to uniform and wm to DTI')
            csfMask = np.logical_and(wm <= 0, gm <= 0)
            output[csfMask] = 0     
            output[gm > 0 ] = 1.0 / ratioDw_Dg # fix gray matter
            borderMask = binary_dilation(csfMask, iterations = 1)
            output[borderMask] = 0

        output[output>upper_limit] = upper_limit
        output[output<0] = 0

        plt.imshow(output[:,:,output.shape[2]//2])
        plt.colorbar()
        plt.title("Before3")
        plt.show()

        return output

    def m_Tildas(self, rgbImg, threshold = 0):
        
        brainmask = np.max(rgbImg, axis = -1) > threshold
        
        retTildas = np.zeros_like(rgbImg)

        for i in range(3):
            retTildas[:,:,:,i] = (np.roll(rgbImg[:,:,:,i],-1,axis=i) + rgbImg[:,:,:,i])/2
            retTildas[:,:,:,i][np.invert(brainmask)] = 0
        
        return retTildas

    def get_D_from_DTI(self, dtiRGB, Dw):
        '''
        dtiRGB: 4D array of shape (Nx,Ny,Nz,3) containing the RGB values of the DTI image
        Dw: diffusion coefficient
        '''
        M = self.m_Tildas(dtiRGB)

        D_minus_x = M[:,:,:,0] * Dw
        D_minus_y = M[:,:,:,1] * Dw
        D_minus_z = M[:,:,:,2] * Dw

        D_plus_x = Dw * np.roll(M[:,:,:,0],1,axis=0)
        D_plus_y = Dw * np.roll(M[:,:,:,1],1,axis=1)
        D_plus_z = Dw * np.roll(M[:,:,:,2],1,axis=2)

        import matplotlib.pyplot as plt
        plt.imshow(D_minus_x[:,:,D_minus_x.shape[2]//2])
        plt.title("D_minus_x")
        plt.colorbar()
        plt.show()

        plt.imshow(D_plus_x[:,:,D_plus_x.shape[2]//2])
        plt.title("D_plus_x")
        plt.colorbar()
        plt.show()

        plt.title("difference")
        plt.imshow(D_plus_x[:,:,D_plus_x.shape[2]//2] - D_minus_x[:,:,D_minus_x.shape[2]//2])
        plt.colorbar()
        plt.show()


        return {"D_minus_x": D_minus_x, "D_minus_y": D_minus_y, "D_minus_z": D_minus_z,"D_plus_x": D_plus_x, "D_plus_y": D_plus_y, "D_plus_z": D_plus_z}
        
    def crop_tissues_and_tumor(self, tissue, tumor_initial, brainmask,  margin=2, threshold=0.0):
        """
        Crop Tissue and tumor_initial such that we remove the maximal amount of voxels
        where the tissue is lower than the threshold.
        A margin is left around the tissues.

        :param Tissue: 4D numpy array of diffusion direction (RGB file)
        :param tumor_initial: 3D numpy array of initial tumor
        :param margin: Margin to leave around the tissues
        :param threshold: Threshold to consider as no tissue
        :return: Cropped tissue, tumor_initial, and the crop coordinates
        """

        # Finding indices where the tissue sum is greater than to the threshold
        tissue_indices = np.argwhere(brainmask > threshold)

        # Finding the bounding box for cropping, considering the margin
        min_coords = np.maximum(tissue_indices.min(axis=0) - margin, 0)
        max_coords = np.minimum(tissue_indices.max(axis=0) + margin + 1, brainmask.shape)

        # Cropping tissue and tumor_initial
        cropped_tissue = tissue[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
    
        cropped_tumor_initial = tumor_initial[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]

        return cropped_tissue, cropped_tumor_initial, (min_coords, max_coords)

    def solve(self, doPlot = False):

        # Unpack parameters
        stopping_time = self.params.get('stopping_time', 100)
        stopping_volume = self.params.get('stopping_volume', np.inf) #mm^3

        Dw = self.params['Dw']
        f = self.params['rho']

        diffusionEllipsoidScaling = self.params['diffusionEllipsoidScaling']
        print(f'diffusionEllipsoidScaling: {diffusionEllipsoidScaling}')

        diffusionTensorExponent = self.params.get('diffusionTensorExponent', 1) # 3 was a good value but 1 is plain linear
        diffusionTensorLinear = self.params.get('diffusionTensorLinear', 0)

        days = self.params.get('days', 100) # Normalized time at 100d, should stay at 100 otherwise it is overparameterized

        NxT1_pct = self.params['NxT1_pct']
        NyT1_pct = self.params['NyT1_pct']
        NzT1_pct = self.params['NzT1_pct']
        res_factor = self.params['resolution_factor']  #Res scaling
        # th_matter = self.params.get('th_matter', 0.1) not yet used
        dx_mm = self.params.get('dx_mm', 1.)  #default 1mm
        dy_mm = self.params.get('dy_mm', 1.)  
        dz_mm = self.params.get('dz_mm', 1.)
        init_scale  = self.params.get('init_scale', 1.)
        time_series_solution_Nt = self.params.get('time_series_solution_Nt', None) #record timeseries, number of steps
        verbose = self.params.get('verbose', False)  

        #print("debug start transform")
        # Apply the transformation
        if diffusionEllipsoidScaling == 1:
            tensor_array_prime = self.params["diffusionTensors"]
        else:
            tensor_array_prime = tools.elongate_tensor_along_main_axis_torch(self.params["diffusionTensors"], diffusionEllipsoidScaling)

        #print("debug end transform")

        if self.params.get('use_homogen_gm', False):
            sGM = self.params['gm']
            sWM = self.params['wm']
            ratioDw_Dg = self.params.get('RatioDw_Dg', 10.)

            # TODO fix tools...
            sRGB = self.makeXYZ_rgb_from_tensor(tensor_array_prime, exponent = diffusionTensorExponent , linear = diffusionTensorLinear, wm = sWM, gm = sGM, ratioDw_Dg = ratioDw_Dg)
        else:
            # TODO fix tools...
            sRGB = self.makeXYZ_rgb_from_tensor(tensor_array_prime, exponent = diffusionTensorExponent , linear = diffusionTensorLinear)

        # Validate input
        assert isinstance(sRGB, np.ndarray), "sRGB must be a numpy array"
        assert sRGB.ndim == 4, "sRGB must be a 4D numpy array, with the last dimension being 3 (RGB)"
        assert 0 <= NxT1_pct <= 1, "NxT1_pct must be between 0 and 1"
        assert 0 <= NyT1_pct <= 1, "NyT1_pct must be between 0 and 1"
        assert 0 <= NzT1_pct <= 1, "NzT1_pct must be between 0 and 1"

        # Interpolate tissue data to lower resolution
        sRGB_low_res = zoom(sRGB, [res_factor, res_factor ,res_factor, 1] , order=1)  # Linear interpolation
        if doPlot:
            from matplotlib import pyplot as plt   
            plotSlice = sRGB[:,:,int(NzT1_pct * sRGB_low_res.shape[2])]
            plt.imshow(plotSlice)# / np.max(plotSlice))
            plt.title('Original - main eigenvector')
            plt.show()
            plt.title('Low res - main eigenvector')
            plotSlice = sRGB_low_res[:,:,int(NzT1_pct * sRGB_low_res.shape[2])]
            plt.imshow(plotSlice / np.max(plotSlice))
            plt.show()

            plt.hist(sRGB[sRGB >0].flatten(), bins=100)
            plt.title('Original - larger then zero')
            plt.show()
            nib.save(nib.Nifti1Image(sRGB_low_res, np.eye(4)), 'sRGB_low_res.nii.gz')
        # Assuming sGM_low_res is already computed using scipy.ndimage.zoom
        original_shape = sRGB_low_res.shape
        new_shape =  sRGB.shape[:3]
        
        # Calculate the zoom factor for each dimension
        extrapolate_factor = tuple(new_sz / float(orig_sz) for new_sz, orig_sz in zip(new_shape, original_shape))

        # Update grid size and steps for low resolution
        Nx, Ny, Nz = sRGB_low_res.shape[:3]

        # Adjust grid steps based on zoom factor
        dx = dx_mm / res_factor
        dy = dy_mm / res_factor
        dz = dz_mm / res_factor

        # Calculate the absolute positions based on percentages
        NxT1 = int(NxT1_pct * Nx)
        NyT1 = int(NyT1_pct * Ny)
        NzT1 = int(NzT1_pct * Nz)

        #stability condition \Delta t \leq \min \left( \frac{\Delta x^2}{6 D_{\text{max}}}, \frac{1}{\rho} \right)
        Nt = np.max([stopping_time * Dw * np.max(sRGB)/np.power((np.min([dx,dy,dz])),2)*8 + 100, stopping_time * f *1.1 ]) 
        dt = stopping_time/Nt
        N_simulation_steps = int(np.ceil(Nt))
        if verbose: 
            print(f'Number of simulation timesteps: {N_simulation_steps}')

        xv, yv, zv = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny), np.arange(0, Nz), indexing='ij')
        A = np.array(self.gauss_sol3d(xv - NxT1, yv - NyT1, zv - NzT1,dx,dy,dz,init_scale))
        print("init: ",A.shape, "Volume of init Tumor", np.sum(A))
        col_res = np.zeros([2, Nx, Ny, Nz])
        col_res[0] = copy.deepcopy(A) #init
        
        #cropping
        brainmask = np.max(sRGB_low_res, axis = -1) > 0.01

        cropped_RGB, A, (min_coords, max_coords) = self.crop_tissues_and_tumor(sRGB_low_res, A, brainmask, margin=2, threshold=0.5)
        
        # Simulation code
        D_domain = self.get_D_from_DTI(cropped_RGB, Dw)

        result = {}
        
        # Initialize time series list if needed
        time_series_data = [] if time_series_solution_Nt is not None else None

        # Determine the steps at which to record the data
        if time_series_data is not None:
            # Using linspace to get exact steps to record, including first and last
            record_steps = np.linspace(0, N_simulation_steps - 1, time_series_solution_Nt, dtype=int)

        #print("debug start simulation")
        try:
            finalTime = None
            result['success'] = False
            
            #check if origin within brainmask
            if not brainmask[NxT1, NyT1, NzT1]:
                raise ValueError("Origin not within brainmask")
            
            for t in range(N_simulation_steps):
                A_Old_size = np.sum(A)
                oldA = copy.deepcopy(A)
                A = self.FK_update(A, D_domain, f, dt, dx, dy, dz)
                #A = np.abs(A)
                volume = dx * dy * dz * np.sum(A)
                if volume >= stopping_volume:
                    finalTime = t * dt
                    break
                
                diffA = np.sum(A) - A_Old_size
                if  diffA < -10:
                    print("Tumor is shrinking at time", t*dt, "by", diffA)
                    result['success'] = False
                    break

                if volume < 0.000001:
                    print("Volume is to small")
                    result['success'] = False
                    break

                if verbose and t % 1000 == 0:
                    imshow_slice = cropped_RGB[:,:,int(NzT1_pct * A.shape[2])]
                    imshow_slice /= np.max(imshow_slice)
                    plt.imshow(imshow_slice)
                    plt.imshow(A[:,:,int(NzT1_pct * A.shape[2])], alpha=0.5*(A[:,:,int(NzT1_pct * A.shape[2])]>0.001), cmap='hot', vmin=0, vmax=1)
                    plt.show()
                    if diffA < 0:
                        print("Tumor is shrinking at time", t*dt, "by", diffA)
                        #plt.imshow(imshow_slice)
                        diffAIMG = np.abs(A - oldA)
                        comz = scipy.ndimage.measurements.center_of_mass(diffAIMG)[2]
                        plt.imshow(diffAIMG[:,:,int(comz)], alpha=0.5*(diffAIMG[:,:,int(comz)]>0.001), cmap='hot')
                        plt.title("Diff")
                        plt.show()

                # Record data at specified steps
                if time_series_data is not None:
                    if t in record_steps:
                        time_series_data.append(copy.deepcopy(A))
            
            if finalTime is None:
                finalTime = stopping_time
            
            # Process final state
            A = self.restore_tumor(sRGB_low_res.shape[:3], A, (min_coords, max_coords))
            col_res[1] = copy.deepcopy(A)  # final

            # Save results in the result dictionary
            result['initial_state'] = np.array(zoom(col_res[0], extrapolate_factor, order=1))
            result['final_state'] = np.array(zoom(col_res[1], extrapolate_factor, order=1))
            result['final_time'] = finalTime
            result['final_volume'] = volume
            result['stopping_criteria'] = 'volume' if volume >= stopping_volume else 'time'
            result['time_series'] = np.array([zoom(self.restore_tumor(sRGB_low_res.shape[:3], state, (min_coords, max_coords)), extrapolate_factor, order=1) for state in time_series_data]) if time_series_data is not None else None
            result['Dw'] = Dw
            result['rho'] = f
            result['success'] = True
            
                    
        except Exception as e:
            print(e)
            result['error'] = str(e)
            result['success'] = False

        return result


# %%
