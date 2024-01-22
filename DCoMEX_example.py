#%%
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors
import time
import scipy.ndimage

import copy
from scipy.ndimage import zoom

class BaseSolver:
    def __init__(self, params):
        self.params = params

    def solve(self):
        raise NotImplementedError("Solve method must be implemented by the subclass.")
    
class FK_Solver(BaseSolver):
    def __init__(self, params):
        super().__init__(params)
    
    def m_Tildas(self, WM,GM,th):
            
        WM_tilda_x = np.where(np.logical_and(np.roll(WM,-1,axis=0) + np.roll(GM,-1,axis=0) >= th,WM + GM >= th),(np.roll(WM,-1,axis=0) + WM)/2,0)
        WM_tilda_y = np.where(np.logical_and(np.roll(WM,-1,axis=1) + np.roll(GM,-1,axis=1) >= th,WM + GM >= th),(np.roll(WM,-1,axis=1) + WM)/2,0)
        WM_tilda_z = np.where(np.logical_and(np.roll(WM,-1,axis=2) + np.roll(GM,-1,axis=2) >= th,WM + GM >= th),(np.roll(WM,-1,axis=2) + WM)/2,0)

        GM_tilda_x = np.where(np.logical_and(np.roll(WM,-1,axis=0) + np.roll(GM,-1,axis=0) >= th,WM + GM >= th),(np.roll(GM,-1,axis=0) + GM)/2,0)
        GM_tilda_y = np.where(np.logical_and(np.roll(WM,-1,axis=1) + np.roll(GM,-1,axis=1) >= th,WM + GM >= th),(np.roll(GM,-1,axis=1) + GM)/2,0)
        GM_tilda_z = np.where(np.logical_and(np.roll(WM,-1,axis=2) + np.roll(GM,-1,axis=2) >= th,WM + GM >= th),(np.roll(GM,-1,axis=2) + GM)/2,0)
        
        return {"WM_t_x": WM_tilda_x,"WM_t_y": WM_tilda_y,"WM_t_z": WM_tilda_z,"GM_t_x": GM_tilda_x,"GM_t_y": GM_tilda_y,"GM_t_z": GM_tilda_z}

    def get_D(self, WM, GM, th, Dw, Dw_ratio):
        M = self.m_Tildas(WM,GM,th)
        D_minus_x = Dw*(M["WM_t_x"] + M["GM_t_x"]/Dw_ratio)
        D_minus_y = Dw*(M["WM_t_y"] + M["GM_t_y"]/Dw_ratio)
        D_minus_z = Dw*(M["WM_t_z"] + M["GM_t_z"]/Dw_ratio)
        
        D_plus_x = Dw*(np.roll(M["WM_t_x"],1,axis=0) + np.roll(M["GM_t_x"],1,axis=0)/Dw_ratio)
        D_plus_y = Dw*(np.roll(M["WM_t_y"],1,axis=1) + np.roll(M["GM_t_y"],1,axis=1)/Dw_ratio)
        D_plus_z = Dw*(np.roll(M["WM_t_z"],1,axis=2) + np.roll(M["GM_t_z"],1,axis=2)/Dw_ratio)
        
        return {"D_minus_x": D_minus_x, "D_minus_y": D_minus_y, "D_minus_z": D_minus_z,"D_plus_x": D_plus_x, "D_plus_y": D_plus_y, "D_plus_z": D_plus_z}

    def FK_update(self, A, D_domain, f, dt, dx, dy, dz):
        D = D_domain
        SP_x = 1/(dx*dx) * (D["D_plus_x"]* (np.roll(A,1,axis=0) - A) - D["D_minus_x"]* (A - np.roll(A,-1,axis=0)) )
        SP_y = 1/(dy*dy) * (D["D_plus_y"]* (np.roll(A,1,axis=1) - A) - D["D_minus_y"]* (A - np.roll(A,-1,axis=1)) )
        SP_z = 1/(dz*dz) * (D["D_plus_z"]* (np.roll(A,1,axis=2) - A) - D["D_minus_z"]* (A - np.roll(A,-1,axis=2)) )
        SP = SP_x + SP_y + SP_z
        diff_A = (SP + f*np.multiply(A,1-A)) * dt
        A += diff_A
        return A

    def crop_tissues_and_tumor(self, GM, WM, tumor_initial, margin=2, threshold=0.05):
        """
        Crop GM, WM, and tumor_initial such that we remove the maximal amount of voxels
        where the sum of GM and WM is lower than the threshold.
        A margin is left around the tissues.

        :param GM: 3D numpy array of gray matter
        :param WM: 3D numpy array of white matter
        :param tumor_initial: 3D numpy array of initial tumor
        :param margin: Margin to leave around the tissues
        :param threshold: Threshold to consider as no tissue
        :return: Cropped GM, WM, tumor_initial, and the crop coordinates
        """

        # Combining GM and WM to find the region with tissue
        tissue_sum = GM + WM

        # Finding indices where the tissue sum is greater than or equal to the threshold
        tissue_indices = np.argwhere(tissue_sum >= threshold)

        # Finding the bounding box for cropping, considering the margin
        min_coords = np.maximum(tissue_indices.min(axis=0) - margin, 0)
        max_coords = np.minimum(tissue_indices.max(axis=0) + margin + 1, GM.shape)

        # Cropping GM, WM, and tumor_initial
        cropped_GM = GM[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
        cropped_WM = WM[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
        cropped_tumor_initial = tumor_initial[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]

        return cropped_GM, cropped_WM, cropped_tumor_initial, (min_coords, max_coords)

    def restore_tumor(self, original_shape, tumor, crop_coords):
        """
        Restore the cropped tumor data back to the original resolution by filling
        the rest of the space with empty voxels.

        :param original_shape: Shape of the original GM/WM arrays
        :param tumor: Cropped tumor 3D numpy array
        :param crop_coords: Coordinates used for cropping (min_coords, max_coords)
        :return: Restored tumor array with original resolution
        """
        restored_tumor = np.zeros(original_shape)
        min_coords, max_coords = crop_coords

        restored_tumor[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]] = tumor

        return restored_tumor

    def gauss_sol3d(self, x, y, z, dx,dy,dz,init_scale):
        # Experimentally chosen
        Dt = 5.0
        M = 250
        
        # Apply scaling to the coordinates
        x_scaled = x * dx/init_scale
        y_scaled = y * dy/init_scale
        z_scaled = z * dz/init_scale

        gauss = M / np.power(4 * np.pi * Dt, 3/2) * np.exp(- (np.power(x_scaled, 2) + np.power(y_scaled, 2) + np.power(z_scaled, 2)) / (4 * Dt))
        gauss = np.where(gauss > 0.1, gauss, 0)
        gauss = np.where(gauss > 1, np.float64(1), gauss)
        return gauss

    def get_initial_configuration(self, Nx,Ny,Nz,NxT,NyT,NzT,r):
        A =  np.zeros([Nx,Ny,Nz])
        if r == 0:
            A[NxT, NyT,NzT] = 1
        else:
            A[NxT-r:NxT+r, NyT-r:NyT+r,NzT-r:NzT+r] = 1
        
        return A

    def solve(self):
        # Unpack parameters
        stopping_time = self.params.get('stopping_time', 100)
        stopping_volume = self.params.get('stopping_volume', np.inf) #mm^3

        Dw = self.params['Dw']
        f = self.params['rho']
        
        sGM = self.params['gm']
        sWM = self.params['wm']
        NxT1_pct = self.params['NxT1_pct']
        NyT1_pct = self.params['NyT1_pct']
        NzT1_pct = self.params['NzT1_pct']
        res_factor = self.params['resolution_factor']  #Res scaling
        RatioDw_Dg = self.params.get('RatioDw_Dg', 10.)
        th_matter = self.params.get('th_matter', 0.1)
        dx_mm = self.params.get('dx_mm', 1.)  #default 1mm
        dy_mm = self.params.get('dy_mm', 1.)  
        dz_mm = self.params.get('dz_mm', 1.)
        init_scale  = self.params.get('init_scale', 1.)
        time_series_solution_Nt = self.params.get('time_series_solution_Nt', None) #record timeseries, number of steps
        verbose = self.params.get('verbose', False)  

        # Validate input
        assert isinstance(sGM, np.ndarray), "sGM must be a numpy array"
        assert isinstance(sWM, np.ndarray), "sWM must be a numpy array"
        assert sGM.ndim == 3, "sGM must be a 3D numpy array"
        assert sWM.ndim == 3, "sWM must be a 3D numpy array"
        assert sGM.shape == sWM.shape
        assert 0 <= NxT1_pct <= 1, "NxT1_pct must be between 0 and 1"
        assert 0 <= NyT1_pct <= 1, "NyT1_pct must be between 0 and 1"
        assert 0 <= NzT1_pct <= 1, "NzT1_pct must be between 0 and 1"

        # Interpolate tissue data to lower resolution
        sGM_low_res = zoom(sGM, res_factor, order=1)  # Linear interpolation
        sWM_low_res = zoom(sWM, res_factor, order=1)
        
        # Assuming sGM_low_res is already computed using scipy.ndimage.zoom
        original_shape = sGM_low_res.shape
        new_shape =  sGM.shape
        
        # Calculate the zoom factor for each dimension
        extrapolate_factor = tuple(new_sz / float(orig_sz) for new_sz, orig_sz in zip(new_shape, original_shape))

        # Update grid size and steps for low resolution
        Nx, Ny, Nz = sGM_low_res.shape


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
        cropped_GM, cropped_WM, A, (min_coords, max_coords) = self.crop_tissues_and_tumor(sGM_low_res, sWM_low_res, A, margin=2, threshold=0.5)
        
        # Simulation code
        D_domain = self.get_D(cropped_WM, cropped_GM, th_matter, Dw, RatioDw_Dg)
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
            A = self.restore_tumor(sGM_low_res.shape, A, (min_coords, max_coords))
            col_res[1] = copy.deepcopy(A)  # final

            # Save results in the result dictionary
            result['initial_state'] = np.array(zoom(col_res[0], extrapolate_factor, order=1))
            result['final_state'] = np.array(zoom(col_res[1], extrapolate_factor, order=1))
            result['final_time'] = finalTime
            result['final_volume'] = volume
            result['stopping_criteria'] = 'volume' if volume >= stopping_volume else 'time'
            result['time_series'] = np.array([zoom(self.restore_tumor(sGM_low_res.shape, state, (min_coords, max_coords)), extrapolate_factor, order=1) for state in time_series_data]) if time_series_data is not None else None
            result['Dw'] = Dw
            result['rho'] = f
            result['success'] = True
            result['NxT1_pct'] = NxT1_pct
            result['NyT1_pct'] = NyT1_pct
            result['NzT1_pct'] = NzT1_pct
                
        except Exception as e:
            result['error'] = str(e)
            result['success'] = False

        return result
    

class Solver_FK_2c(FK_Solver):
    def __init__(self, params):
        super().__init__(params)

    def m_Tildas_with_necro(self, WM, GM, PC, PN, th, th_necro):
        # Helper function to create combined conditions for each axis
        def combined_condition(axis):
            matter_cond = np.logical_and(np.roll(WM + GM, -1, axis=axis) >= th, WM + GM >= th)
            full_region_cond = np.logical_and(np.roll(PC + PN, -1, axis=axis) <= th_necro, PC + PN <= th_necro)
            return np.logical_and(matter_cond, full_region_cond)

        # Combined conditions for x, y, z axes
        combined_cond_x = combined_condition(0)
        combined_cond_y = combined_condition(1)
        combined_cond_z = combined_condition(2)

        # Calculate tildas using combined conditions
        WM_tilda_x = np.where(combined_cond_x, (np.roll(WM, -1, axis=0) + WM) / 2, 0)
        WM_tilda_y = np.where(combined_cond_y, (np.roll(WM, -1, axis=1) + WM) / 2, 0)
        WM_tilda_z = np.where(combined_cond_z, (np.roll(WM, -1, axis=2) + WM) / 2, 0)

        GM_tilda_x = np.where(combined_cond_x, (np.roll(GM, -1, axis=0) + GM) / 2, 0)
        GM_tilda_y = np.where(combined_cond_y, (np.roll(GM, -1, axis=1) + GM) / 2, 0)
        GM_tilda_z = np.where(combined_cond_z, (np.roll(GM, -1, axis=2) + GM) / 2, 0)

        return {
            "WM_t_x": WM_tilda_x, "WM_t_y": WM_tilda_y, "WM_t_z": WM_tilda_z,
            "GM_t_x": GM_tilda_x, "GM_t_y": GM_tilda_y, "GM_t_z": GM_tilda_z
        }

    def get_D_with_necro(self, WM,GM,th,Dw,Dw_ratio, PC, NC, th_necro):
        M = self.m_Tildas_with_necro(WM,GM,PC,NC,th, th_necro)
        D_minus_x = Dw*(M["WM_t_x"] + M["GM_t_x"]/Dw_ratio)
        D_minus_y = Dw*(M["WM_t_y"] + M["GM_t_y"]/Dw_ratio)
        D_minus_z = Dw*(M["WM_t_z"] + M["GM_t_z"]/Dw_ratio)
        
        D_plus_x = Dw*(np.roll(M["WM_t_x"],1,axis=0) + np.roll(M["GM_t_x"],1,axis=0)/Dw_ratio)
        D_plus_y = Dw*(np.roll(M["WM_t_y"],1,axis=1) + np.roll(M["GM_t_y"],1,axis=1)/Dw_ratio)
        D_plus_z = Dw*(np.roll(M["WM_t_z"],1,axis=2) + np.roll(M["GM_t_z"],1,axis=2)/Dw_ratio)
        
        return {"D_minus_x": D_minus_x, "D_minus_y": D_minus_y, "D_minus_z": D_minus_z,"D_plus_x": D_plus_x, "D_plus_y": D_plus_y, "D_plus_z": D_plus_z}

    def FK_2c_update(self, P, N, S, D_domain, f, lambda_np, sigma_np, D_s_domain, lambda_s, dt, dx, dy, dz):
        D = D_domain
        D_s = D_s_domain
        H = self.smooth_heaviside(S, sigma_np)

        # Update for P (proliferative cells)
        SP_x = 1/(dx*dx) * (D["D_plus_x"] * (np.roll(P, 1, axis=0) - P) - D["D_minus_x"] * (P - np.roll(P, -1, axis=0)))
        SP_y = 1/(dy*dy) * (D["D_plus_y"] * (np.roll(P, 1, axis=1) - P) - D["D_minus_y"] * (P - np.roll(P, -1, axis=1)))
        SP_z = 1/(dz*dz) * (D["D_plus_z"] * (np.roll(P, 1, axis=2) - P) - D["D_minus_z"] * (P - np.roll(P, -1, axis=2)))
        SP = SP_x + SP_y + SP_z
        diff_P = (SP + f * np.multiply(S, P) * (1 - P - N) - lambda_np * P * H) * dt
        P += diff_P

        # Update for N (necrotic cells)
        diff_N = lambda_np * P * H * dt
        N += diff_N

        # Update for S (nutrient field)
        SS_x = 1/(dx*dx) * (D_s["D_plus_x"] * (np.roll(S, 1, axis=0) - S) - D_s["D_minus_x"] * (S - np.roll(S, -1, axis=0)))
        SS_y = 1/(dy*dy) * (D_s["D_plus_y"] * (np.roll(S, 1, axis=1) - S) - D_s["D_minus_y"] * (S - np.roll(S, -1, axis=1)))
        SS_z = 1/(dz*dz) * (D_s["D_plus_z"] * (np.roll(S, 1, axis=2) - S) - D_s["D_minus_z"] * (S - np.roll(S, -1, axis=2)))
        SS = SS_x + SS_y + SS_z
        diff_S = (SS - lambda_s * S * P) * dt
        S += diff_S

        return {'P': P, 'N': N, 'S': S}

    def smooth_heaviside(self, x, sigma_np, k=50):
        return 1- 1 / (1 + np.exp(-k * (x - sigma_np)))


    def crop_tissues_and_tumor(self, GM, WM, tumor_initial, necrotic_initial, nutrient_field, margin=2, threshold=0.1):
        """
        Crop GM, WM, tumor_initial, necrotic_initial, and nutrient_field such that we remove the maximal amount of voxels
        where the sum of GM and WM is lower than the threshold. A margin is left around the tissues.

        :param GM: 3D numpy array of gray matter
        :param WM: 3D numpy array of white matter
        :param tumor_initial: 3D numpy array of initial tumor
        :param necrotic_initial: 3D numpy array of initial necrotic cells
        :param nutrient_field: 3D numpy array of nutrient field
        :param margin: Margin to leave around the tissues
        :param threshold: Threshold to consider as no tissue
        :return: Cropped GM, WM, tumor_initial, necrotic_initial, nutrient_field, and the crop coordinates
        """

        # Combining GM and WM to find the region with tissue
        tissue_sum = GM + WM

        # Finding indices where the tissue sum is greater than or equal to the threshold
        tissue_indices = np.argwhere(tissue_sum >= threshold)

        # Finding the bounding box for cropping, considering the margin
        min_coords = np.maximum(tissue_indices.min(axis=0) - margin, 0)
        max_coords = np.minimum(tissue_indices.max(axis=0) + margin + 1, GM.shape)

        # Function to crop a given array
        def crop_array(arr):
            return arr[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]

        # Cropping GM, WM, tumor_initial, necrotic_initial, and nutrient_field
        cropped_GM = crop_array(GM)
        cropped_WM = crop_array(WM)
        
        cropped_states = {
            'P': crop_array(tumor_initial),
            'N': crop_array(necrotic_initial),
            'S': crop_array(nutrient_field)
        }

        return cropped_GM, cropped_WM, cropped_states, (min_coords, max_coords)


    def get_initial_configuration(self, NxT,NyT,NzT,Nx,Ny,Nz,dx,dy,dz,init_scale,GM,WM,th_matter):
        xv, yv, zv = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny), np.arange(0, Nz), indexing='ij')
        P = np.array(self.gauss_sol3d(xv - NxT, yv - NyT, zv - NzT,dx,dy,dz,init_scale))
        N = np.zeros(P.shape)
        S = np.ones(P.shape)
        #remove csf from S
        S = np.where(WM + GM >= th_matter, S, 0)
        
        initial_states = {'P': P, 'N': N, 'S': S}
        
        return initial_states
    

    def solve(self):
        # Unpack parameters
        stopping_time = self.params.get('stopping_time', 100)
        stopping_volume = self.params.get('stopping_volume', np.inf) #mm^3

        Dw = self.params['Dw']
        f = self.params['rho']
        lambda_np = self.params['lambda_np']
        sigma_np = self.params['sigma_np']
        D_s = self.params['D_s']
        lambda_s =self.params['lambda_s']
        
        sGM = self.params['gm']
        sWM = self.params['wm']
        NxT1_pct = self.params['NxT1_pct']
        NyT1_pct = self.params['NyT1_pct']
        NzT1_pct = self.params['NzT1_pct']
        res_factor = self.params['resolution_factor']  #Res scaling
        RatioDw_Dg = self.params.get('RatioDw_Dg', 10.)
        th_matter = self.params.get('th_matter', 0.1)  #when to stop diffusing to a region: when matter <= 0.1
        th_necro = self.params.get('th_necro', 0.9) #when to stop diffusing to a region: when cells >= 0.9
        dx_mm = self.params.get('dx_mm', 1.)  #default 1mm
        dy_mm = self.params.get('dy_mm', 1.)  
        dz_mm = self.params.get('dz_mm', 1.)
        init_scale  = self.params.get('init_scale', 1.)
        time_series_solution_Nt = self.params.get('time_series_solution_Nt', None) #record timeseries, number of steps
        Nt_multiplier = self.params.get('Nt_multiplier',8)
        verbose = self.params.get('verbose', False)  

        # Validate input
        assert isinstance(sGM, np.ndarray), "sGM must be a numpy array"
        assert isinstance(sWM, np.ndarray), "sWM must be a numpy array"
        assert sGM.ndim == 3, "sGM must be a 3D numpy array"
        assert sWM.ndim == 3, "sWM must be a 3D numpy array"
        assert sGM.shape == sWM.shape
        assert 0 <= NxT1_pct <= 1, "NxT1_pct must be between 0 and 1"
        assert 0 <= NyT1_pct <= 1, "NyT1_pct must be between 0 and 1"
        assert 0 <= NzT1_pct <= 1, "NzT1_pct must be between 0 and 1"

        # Interpolate tissue data to lower resolution
        sGM_low_res = zoom(sGM, res_factor, order=1)  # Linear interpolation
        sWM_low_res = zoom(sWM, res_factor, order=1)
        
        # Assuming sGM_low_res is already computed using scipy.ndimage.zoom
        original_shape = sGM_low_res.shape
        new_shape =  sGM.shape
        
        # Calculate the zoom factor for each dimension
        extrapolate_factor = tuple(new_sz / float(orig_sz) for new_sz, orig_sz in zip(new_shape, original_shape))

        # Update grid size and steps for low resolution
        Nx, Ny, Nz = sGM_low_res.shape


        # Adjust grid steps based on zoom factor
        dx = dx_mm / res_factor
        dy = dy_mm / res_factor
        dz = dz_mm / res_factor


        # Calculate the absolute positions based on percentages
        NxT1 = int(NxT1_pct * Nx)
        NyT1 = int(NyT1_pct * Ny)
        NzT1 = int(NzT1_pct * Nz)
        Nt = stopping_time * np.max([Dw,D_s])/np.power((np.min([dx,dy,dz])),2)*Nt_multiplier + 100
        dt = stopping_time/Nt
        N_simulation_steps = int(np.ceil(Nt))
        if verbose: 
            print(f'Number of simulation timesteps: {N_simulation_steps}')

        # Assuming get_initial_configuration now returns a dictionary
        initial_states = self.get_initial_configuration(NxT1, NyT1, NzT1, Nx, Ny, Nz, dx, dy, dz, init_scale, sGM_low_res, sWM_low_res, th_matter)

        col_res = {'initial_state': {}, 'final_state': {}}
        col_res['initial_state'] = copy.deepcopy(initial_states)  # Store initial states

        # Cropping
        cropped_GM, cropped_WM, cropped_states, (min_coords, max_coords) = self.crop_tissues_and_tumor(sGM_low_res, sWM_low_res, initial_states['P'], initial_states['N'], initial_states['S'], margin=2, threshold=0.5)
        # Simulation code
        D_domain = self.get_D_with_necro(cropped_WM, cropped_GM, th_matter, Dw, RatioDw_Dg, cropped_states['P'],cropped_states['N'],th_necro)
        D_s_domain = self.get_D(cropped_WM, cropped_GM, th_matter, D_s, 1)
        result = {}

        if time_series_solution_Nt is not None:
            time_series_data = {'P': [], 'N': [], 'S': []}

        # Determine the steps at which to record the data
        if time_series_solution_Nt is not None:
            record_steps = np.linspace(0, N_simulation_steps - 1, time_series_solution_Nt, dtype=int)

        #try:
        finalTime = None
        for t in range(N_simulation_steps):
            updated_states = self.FK_2c_update(cropped_states['P'], cropped_states['N'], cropped_states['S'], D_domain, f,lambda_np, sigma_np, D_s_domain, lambda_s,  dt, dx, dy, dz)
            D_domain = self.get_D_with_necro(cropped_WM, cropped_GM, th_matter, Dw, RatioDw_Dg, cropped_states['P'], updated_states['N'],th_necro)
            # Update states
            cropped_states.update(updated_states)

            # use volume of proliferative and necrotic cells
            volume = dx * dy * dz * np.sum(updated_states['P']) +np.sum(updated_states['N'])
            if volume >= stopping_volume:
                finalTime = t * dt
                break

            # Record data at specified steps
            if time_series_solution_Nt is not None and t in record_steps:
                for field in ['P', 'N', 'S']:
                    time_series_data[field].append(copy.deepcopy(cropped_states[field]))
        if finalTime is None:
            finalTime = stopping_time

        # Process final state
        for key in ['P', 'N', 'S']:
            restored_field = self.restore_tumor(sGM_low_res.shape, cropped_states[key], (min_coords, max_coords))
            col_res['final_state'][key] = copy.deepcopy(restored_field)  # Store final states

        # Save results in the result dictionary
        result['initial_state'] = {k: np.array(zoom(v, extrapolate_factor, order=1)) for k, v in col_res['initial_state'].items()}
        result['final_state'] = {k: np.array(zoom(v, extrapolate_factor, order=1)) for k, v in col_res['final_state'].items()}
        # Process and store the time series data
        if time_series_solution_Nt is not None:
            # Convert lists to arrays
            for field in ['P', 'N', 'S']:
                time_series_data[field] = np.array(time_series_data[field])

            result['time_series'] = {field: [zoom(self.restore_tumor(sGM_low_res.shape, state, (min_coords, max_coords)), extrapolate_factor, order=1) for state in time_series_data[field]] for field in ['P', 'N', 'S']}

        else:
            result['time_series'] = None
        result['Dw'] = Dw
        result['rho'] = f
        result['success'] = True
        result['NxT1_pct'] = NxT1_pct
        result['NyT1_pct'] = NyT1_pct
        result['NzT1_pct'] = NzT1_pct
        result['final_time'] = finalTime
        result['final_volume'] = volume
        result['stopping_criteria'] = 'volume' if volume >= stopping_volume else 'time'

        #except Exception as e:
        #    result['error'] = str(e)
        #    result['success'] = False

        return result

# Plotting function
def plot_tumor_states(wm_data, initial_states, final_states, slice_index, cmap1, cmap2, cmap3):
    plt.figure(figsize=(18, 6))  # Adjusted figure size for 3 columns

    # Fields to plot
    fields = ['P', 'N', 'S']
    titles = ['Proliferative Cells', 'Necrotic Cells', 'Nutrient Field']

    # Plotting initial states
    for i, field in enumerate(fields):
        plt.subplot(2, 3, i + 1)
        plt.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
        plt.imshow(initial_states[field][:, :, slice_index], cmap=cmap2 if i == 0 else cmap3, vmin=0, vmax=1, alpha=0.65)
        plt.title(f"Initial {titles[i]}")

    # Plotting final states
    for i, field in enumerate(fields):
        plt.subplot(2, 3, i + 4)
        plt.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
        plt.imshow(final_states[field][:, :, slice_index], cmap=cmap2 if i == 0 else cmap3, vmin=0, vmax=1, alpha=0.65)
        plt.title(f"Final {titles[i]}")

    plt.tight_layout()
    plt.show()

def plot_time_series(gm_data, wm_data, time_series_data, slice_index, cmap1, cmap2):
    plt.figure(figsize=(8, 24))

    # Fields to plot
    fields = ['P', 'N', 'S']
    field_titles = ['Proliferative', 'Necrotic', 'Nutrient']

    # Generate indices for selected timesteps
    num_timesteps = np.array(time_series_data['P']).shape[0]
    time_points = np.linspace(0, num_timesteps - 1, 8, dtype=int)
    time_max = num_timesteps - 1
    th_plot = 0.1
    margin_x = 25
    margin_y = 25
    for i, t in enumerate(time_points):
        # Calculate the relative time (0 to 1)
        relative_time = t / time_max

        for j, field in enumerate(fields):
            ax = plt.subplot(len(time_points), 3, i * 3 + j + 1)

            # Plot the white matter data
            plt.contourf(np.fliplr(np.flipud(np.rot90(gm_data[margin_x:-margin_x, margin_y:-margin_y, slice_index], -1))), levels=[0.5, 1], colors='gray', alpha=0.35)
            # Plot the field data
            vol = np.array(time_series_data[field])[t,margin_x:-margin_x, margin_y:-margin_y, slice_index]
            vol_display = np.array(np.where(vol > th_plot, vol, np.nan))
            plt.imshow(np.fliplr(np.flipud(np.rot90(vol_display,-1))), cmap=cmap2, vmin=0, vmax=1, alpha=1.)

            # Add field titles
            plt.title(f"{field_titles[j]}")

            # Add time annotation only once per row, above the subplot
            if j == 0:
                ax.text(0.5, 1.20, f"Time: {relative_time:.2f}", transform=ax.transAxes, fontsize=12, fontweight='bold', ha='center', va='center')
                
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.show()
#%%  #################################### start here settings ####################################
# Create binary segmentation masks
gm_data = nib.load('dataset/GM.nii.gz').get_fdata()
wm_data = nib.load('dataset/WM.nii.gz').get_fdata()
affine = nib.load('dataset/GM.nii.gz').affine

# Set up parameters
parameters = {
    'Dw': 0.9,          # Diffusion coefficient for the white matter
    'rho': 0.14,         # Proliferation rate
    'lambda_np': 0.35, # Transition rate between proli and necrotic cells
    'sigma_np': 0.5, #Transition threshols between proli and necrotic given nutrient field
    'D_s': 1.3,      # Diffusion coefficient for the nutrient field
    'lambda_s': 0.05, # Proli cells nutrients consumption rate
    'RatioDw_Dg': 100,  # Ratio of diffusion coefficients in white and grey matter
    'Nt_multiplier': 8,
    'gm': gm_data,      # Grey matter data
    'wm': wm_data,      # White matter data
    'NxT1_pct': 0.35,    # tumor position [%]
    'NyT1_pct': 0.6,
    'NzT1_pct': 0.5,
    'init_scale': 1., #scale of the initial gaussian
    'resolution_factor': 0.5, #resultion scaling for calculations
    'th_matter': 0.1, #when to stop diffusing: at th_matter > gm+wm
    'verbose': True, #printing timesteps 
    'time_series_solution_Nt': 8, # number of timesteps in the output
    'stopping_volume': 10000
}

###################################### end here settings ####################################
# Run the FK_solver and plot the results
start_time = time.time()
fk_solver = Solver_FK_2c(parameters)
result = fk_solver.solve()
end_time = time.time()  # Store the end time
execution_time = int(end_time - start_time)  # Calculate the difference
print(f"Execution Time: {execution_time} seconds")

#save results
nib.save(nib.Nifti1Image(result['final_state']['P'], affine), './tumor_final.nii.gz')
nib.save(nib.Nifti1Image(result['final_state']['N'], affine), './necrotic_final.nii.gz')
nib.save(nib.Nifti1Image(result['final_state']['S'],affine), './nutrient_final.nii.gz')


#%% plotting
# Calculate the slice index
NzT = int(parameters['NzT1_pct'] * gm_data.shape[2])
# Create custom color maps
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['black', 'white'], 256)
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap2', ['black', 'green', 'yellow', 'red'], 256)

if result['success']:
    print("Simulation successful!")
    # Extract initial and final states from the result
    #initial_states = result['initial_state']
    #final_states = result['final_state']
    #plot_tumor_states(wm_data, initial_states, final_states, NzT, cmap1, cmap2, cmap2)
    time_series_data = result['time_series']
    plot_time_series(gm_data, wm_data, time_series_data, NzT, cmap1, cmap2)
else:
    print("Error occurred:", result['error'])

# %%
print("finalVolume in mm^3", result['final_volume'])
print("finalTime", result['final_time'])
print("stopping criteria", result['stopping_criteria'])