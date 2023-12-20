import numpy as np
import copy
from scipy.ndimage import zoom
from ..base_solver import BaseSolver

def m_Tildas(WM,GM,th):
        
    WM_tilda_x = np.where(np.logical_and(np.roll(WM,-1,axis=0) + np.roll(GM,-1,axis=0) >= th,WM + GM >= th),(np.roll(WM,-1,axis=0) + WM)/2,0)
    WM_tilda_y = np.where(np.logical_and(np.roll(WM,-1,axis=1) + np.roll(GM,-1,axis=1) >= th,WM + GM >= th),(np.roll(WM,-1,axis=1) + WM)/2,0)
    WM_tilda_z = np.where(np.logical_and(np.roll(WM,-1,axis=2) + np.roll(GM,-1,axis=2) >= th,WM + GM >= th),(np.roll(WM,-1,axis=2) + WM)/2,0)

    GM_tilda_x = np.where(np.logical_and(np.roll(WM,-1,axis=0) + np.roll(GM,-1,axis=0) >= th,WM + GM >= th),(np.roll(GM,-1,axis=0) + GM)/2,0)
    GM_tilda_y = np.where(np.logical_and(np.roll(WM,-1,axis=1) + np.roll(GM,-1,axis=1) >= th,WM + GM >= th),(np.roll(GM,-1,axis=1) + GM)/2,0)
    GM_tilda_z = np.where(np.logical_and(np.roll(WM,-1,axis=2) + np.roll(GM,-1,axis=2) >= th,WM + GM >= th),(np.roll(GM,-1,axis=2) + GM)/2,0)
    
    return {"WM_t_x": WM_tilda_x,"WM_t_y": WM_tilda_y,"WM_t_z": WM_tilda_z,"GM_t_x": GM_tilda_x,"GM_t_y": GM_tilda_y,"GM_t_z": GM_tilda_z}

def get_D(WM,GM,th,Dw,Dw_ratio):
    M = m_Tildas(WM,GM,th)
    D_minus_x = Dw*(M["WM_t_x"] + M["GM_t_x"]/Dw_ratio)
    D_minus_y = Dw*(M["WM_t_y"] + M["GM_t_y"]/Dw_ratio)
    D_minus_z = Dw*(M["WM_t_y"] + M["GM_t_y"]/Dw_ratio)
    
    D_plus_x = Dw*(np.roll(M["WM_t_x"],1,axis=0) + np.roll(M["GM_t_x"],1,axis=0)/Dw_ratio)
    D_plus_y = Dw*(np.roll(M["WM_t_y"],1,axis=1) + np.roll(M["GM_t_y"],1,axis=1)/Dw_ratio)
    D_plus_z = Dw*(np.roll(M["WM_t_z"],1,axis=2) + np.roll(M["GM_t_z"],1,axis=2)/Dw_ratio)
    
    return {"D_minus_x": D_minus_x, "D_minus_y": D_minus_y, "D_minus_z": D_minus_z,"D_plus_x": D_plus_x, "D_plus_y": D_plus_y, "D_plus_z": D_plus_z}

def FK_2c_update(P, N, S, D_domain, f, lambda_np, sigma_np, D_s, lambda_s, dt, dx, dy, dz):
    D = D_domain
    H = smooth_heaviside(S, sigma_np)

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
    SS_x = 1/(dx*dx) * (D_s * (np.roll(S, 1, axis=0) - 2 * S + np.roll(S, -1, axis=0)))
    SS_y = 1/(dy*dy) * (D_s * (np.roll(S, 1, axis=1) - 2 * S + np.roll(S, -1, axis=1)))
    SS_z = 1/(dz*dz) * (D_s * (np.roll(S, 1, axis=2) - 2 * S + np.roll(S, -1, axis=2)))
    SS = SS_x + SS_y + SS_z
    diff_S = (SS - lambda_s * S * P) * dt
    S += diff_S

    return {'P': P, 'N': N, 'S': S}

def smooth_heaviside(x, sigma_np, k=50):
    return 1- 1 / (1 + np.exp(-k * (x - sigma_np)))


def crop_tissues_and_tumor(GM, WM, tumor_initial, necrotic_initial, nutrient_field, margin=2, threshold=0.1):
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



def restore_tumor(original_shape, tumor, crop_coords):
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


def restore_tumor(original_shape, tumor, crop_coords):
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


def gauss_sol3d(x, y, z, dx,dy,dz,init_scale):
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




def get_initial_configuration(NxT,NyT,NzT,Nx,Ny,Nz,dx,dy,dz,init_scale):
    xv, yv, zv = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny), np.arange(0, Nz), indexing='ij')
    P = np.array(gauss_sol3d(xv - NxT, yv - NyT, zv - NzT,dx,dy,dz,init_scale))
    N = np.zeros(P.shape)
    S = np.ones(P.shape)
    
    initial_states = {'P': P, 'N': N, 'S': S}
    
    return initial_states

class Solver(BaseSolver):
    def __init__(self, params):
        super().__init__(params)
    def solve(self):
        # Unpack parameters
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
        th_matter = self.params.get('th_matter', 0.1)
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
        days = 100
        Nt = days * Dw/np.power((np.min([dx,dy,dz])),2)*Nt_multiplier + 100
        dt = days/Nt
        N_simulation_steps = int(np.ceil(Nt))
        if verbose: 
            print(f'Number of simulation timesteps: {N_simulation_steps}')

        # Assuming get_initial_configuration now returns a dictionary
        initial_states = get_initial_configuration(NxT1, NyT1, NzT1, Nx, Ny, Nz, dx, dy, dz, init_scale)

        col_res = {'initial_state': {}, 'final_state': {}}
        col_res['initial_state'] = copy.deepcopy(initial_states)  # Store initial states

        # Cropping
        cropped_GM, cropped_WM, cropped_states, (min_coords, max_coords) = crop_tissues_and_tumor(sGM_low_res, sWM_low_res, initial_states['P'], initial_states['N'], initial_states['S'], margin=2, threshold=0.5)
        # Simulation code
        D_domain = get_D(cropped_WM, cropped_GM, th_matter, Dw, RatioDw_Dg)
        result = {}

        if time_series_solution_Nt is not None:
            time_series_data = {'P': [], 'N': [], 'S': []}

        # Determine the steps at which to record the data
        if time_series_solution_Nt is not None:
            record_steps = np.linspace(0, N_simulation_steps - 1, time_series_solution_Nt, dtype=int)

        #try:
        for t in range(N_simulation_steps):
            updated_states = FK_2c_update(cropped_states['P'], cropped_states['N'], cropped_states['S'], D_domain, f,lambda_np, sigma_np, D_s, lambda_s,  dt, dx, dy, dz)

            # Update states
            cropped_states.update(updated_states)

            # Record data at specified steps
            if time_series_solution_Nt is not None and t in record_steps:
                for field in ['P', 'N', 'S']:
                    time_series_data[field].append(copy.deepcopy(cropped_states[field]))

        # Process final state
        for key in ['P', 'N', 'S']:
            restored_field = restore_tumor(sGM_low_res.shape, cropped_states[key], (min_coords, max_coords))
            col_res['final_state'][key] = copy.deepcopy(restored_field)  # Store final states

        # Save results in the result dictionary
        result['initial_state'] = {k: np.array(zoom(v, extrapolate_factor, order=1)) for k, v in col_res['initial_state'].items()}
        result['final_state'] = {k: np.array(zoom(v, extrapolate_factor, order=1)) for k, v in col_res['final_state'].items()}
        # Process and store the time series data
        if time_series_solution_Nt is not None:
            # Convert lists to arrays
            for field in ['P', 'N', 'S']:
                time_series_data[field] = np.array(time_series_data[field])

            result['time_series'] = {field: [zoom(state, extrapolate_factor, order=1) for state in time_series_data[field]] for field in ['P', 'N', 'S']}

        else:
            result['time_series'] = None
        result['Dw'] = Dw
        result['rho'] = f
        result['success'] = True
        result['NxT1_pct'] = NxT1_pct
        result['NyT1_pct'] = NyT1_pct
        result['NzT1_pct'] = NzT1_pct

        #except Exception as e:
        #    result['error'] = str(e)
        #    result['success'] = False

        return result
