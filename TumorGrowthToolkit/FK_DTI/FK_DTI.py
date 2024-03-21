#%%
import numpy as np
import copy
from scipy.ndimage import zoom
from ..FK.FK import Solver as FK_Solver

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

    def get_D_from_DTI(self, dtiRGB, DwMax, exponent = 1 , maskOut=None, linear = 0):
        '''
        dtiRGB: 4D array of shape (Nx,Ny,Nz,3) containing the RGB values of the DTI image
        DwMax: maximum diffusion coefficient
        
        # monotonic function to map RGB to diffusion coefficient
        exponent: an exponent to increase/decrease the impact of large/small RGB values
        linear: a linear factor to make the function more or less steep

        maskOut: a mask to mask out certain regions of the brain. If None, no mask is applied
        '''
        if maskOut is not None:
            dtiRGB[maskOut] = 0
        
        D_minus_x = (dtiRGB[:,:,:,0]**exponent + dtiRGB[:,:,:,0] * linear) *DwMax
        D_minus_y = (dtiRGB[:,:,:,1]**exponent + dtiRGB[:,:,:,1] * linear) *DwMax
        D_minus_z = (dtiRGB[:,:,:,2]**exponent + dtiRGB[:,:,:,2] * linear) *DwMax

        D_plus_x = D_minus_x
        D_plus_y = D_minus_y
        D_plus_z = D_minus_z

        return {"D_minus_x": D_minus_x, "D_minus_y": D_minus_y, "D_minus_z": D_minus_z,"D_plus_x": D_plus_x, "D_plus_y": D_plus_y, "D_plus_z": D_plus_z}
        
    def crop_tissues_and_tumor(self, tissue, tumor_initial, margin=2, threshold=0.0):
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
        tissue_indices = np.argwhere(tissue > threshold)

        # Finding the bounding box for cropping, considering the margin
        min_coords = np.maximum(tissue_indices.min(axis=0) - margin, 0)
        max_coords = np.minimum(tissue_indices.max(axis=0) + margin + 1, tissue.shape)

        # Cropping tissue and tumor_initial
        cropped_tissue = tissue[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
    
        cropped_tumor_initial = tumor_initial[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]

        return cropped_tissue, cropped_tumor_initial, (min_coords, max_coords)

    def solve(self):

        # Unpack parameters
        stopping_time = self.params.get('stopping_time', 100)
        stopping_volume = self.params.get('stopping_volume', np.inf) #mm^3

        Dw = self.params['Dw']
        f = self.params['rho']
        
        sRGB = self.params['rgb']

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

        # Validate input
        assert isinstance(sRGB, np.ndarray), "sRGB must be a numpy array"
        assert sRGB.ndim == 4, "sRGB must be a 4D numpy array, with the last dimension being 3 (RGB)"
        assert 0 <= NxT1_pct <= 1, "NxT1_pct must be between 0 and 1"
        assert 0 <= NyT1_pct <= 1, "NyT1_pct must be between 0 and 1"
        assert 0 <= NzT1_pct <= 1, "NzT1_pct must be between 0 and 1"

        # Interpolate tissue data to lower resolution
        sRGB_low_res = zoom(sRGB, [res_factor, res_factor ,res_factor, 1] , order=1)  # Linear interpolation
        
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

        Nt = stopping_time * Dw/np.power((np.min([dx,dy,dz])),2)*8 + 100
        dt = stopping_time/Nt
        N_simulation_steps = int(np.ceil(Nt))
        if verbose: 
            print(f'Number of simulation timesteps: {N_simulation_steps}')

        xv, yv, zv = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny), np.arange(0, Nz), indexing='ij')
        A = np.array(self.gauss_sol3d(xv - NxT1, yv - NyT1, zv - NzT1,dx,dy,dz,init_scale))
        col_res = np.zeros([2, Nx, Ny, Nz])
        col_res[0] = copy.deepcopy(A) #init
        
        #cropping
        cropped_RGB, A, (min_coords, max_coords) = self.crop_tissues_and_tumor(sRGB_low_res, A, margin=2, threshold=0.5)
        
        # Simulation code
        D_domain = self.get_D_from_DTI(cropped_RGB, Dw, exponent = diffusionTensorExponent , linear = diffusionTensorLinear)

        result = {}
        
        # Initialize time series list if needed
        time_series_data = [] if time_series_solution_Nt is not None else None

        # Determine the steps at which to record the data
        if time_series_data is not None:
            # Using linspace to get exact steps to record, including first and last
            record_steps = np.linspace(0, N_simulation_steps - 1, time_series_solution_Nt, dtype=int)

        try:
            finalTime = None
            for t in range(N_simulation_steps):
                A = self.FK_update(A, D_domain, f, dt, dx, dy, dz)

                volume = dx * dy * dz * np.sum(A)
                if volume >= stopping_volume:
                    finalTime = t * dt
                    break

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
            result['error'] = str(e)
            result['success'] = False

        return result

