import numpy as np
import copy
from scipy.ndimage import zoom
from ..base_solver import BaseSolver

from ..FK.FK import Solver as FK_Solver

class FK_DTI_Solver(FK_Solver):
    def __init__(self, params):
        super().__init__(params)
    
    def solve(self):
        return super().solve()
        '''
        # Unpack parameters
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
        days = 100
        Nt = days * Dw/np.power((np.min([dx,dy,dz])),2)*8 + 100
        dt = days/Nt
        N_simulation_steps = int(np.ceil(Nt))
        if verbose: 
            print(f'Number of simulation timesteps: {N_simulation_steps}')

        xv, yv, zv = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny), np.arange(0, Nz), indexing='ij')
        A = np.array(gauss_sol3d(xv - NxT1, yv - NyT1, zv - NzT1,dx,dy,dz,init_scale))
        col_res = np.zeros([2, Nx, Ny, Nz])
        col_res[0] = copy.deepcopy(A) #init
        
        #cropping
        cropped_GM, cropped_WM, A, (min_coords, max_coords) = crop_tissues_and_tumor(sGM_low_res, sWM_low_res, A, margin=2, threshold=0.5)
        
        # Simulation code
        D_domain = get_D(cropped_WM, cropped_GM, th_matter, Dw, RatioDw_Dg)
        result = {}
        
        # Initialize time series list if needed
        time_series_data = [] if time_series_solution_Nt is not None else None

        # Determine the steps at which to record the data
        if time_series_data is not None:
            # Using linspace to get exact steps to record, including first and last
            record_steps = np.linspace(0, N_simulation_steps - 1, time_series_solution_Nt, dtype=int)

        try:
            for t in range(N_simulation_steps):
                A = FK_update(A, D_domain, f, dt, dx, dy, dz)

                # Record data at specified steps
                if time_series_data is not None:
                    if t in record_steps:
                        time_series_data.append(copy.deepcopy(A))

            # Process final state
            A = restore_tumor(sGM_low_res.shape, A, (min_coords, max_coords))
            col_res[1] = copy.deepcopy(A)  # final

            # Save results in the result dictionary
            result['initial_state'] = np.array(zoom(col_res[0], extrapolate_factor, order=1))
            result['final_state'] = np.array(zoom(col_res[1], extrapolate_factor, order=1))
            result['time_series'] = np.array([zoom(restore_tumor(sGM_low_res.shape, state, (min_coords, max_coords)), extrapolate_factor, order=1) for state in time_series_data]) if time_series_data is not None else None
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
        '''
