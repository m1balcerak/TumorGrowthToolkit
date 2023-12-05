import numpy as np
import copy

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

def FK_update(A,D_domain,f, dt,dx,dy,dz):
    D = D_domain
    SP_x = 1/(dx*dx) * (D["D_plus_x"]* (np.roll(A,1,axis=0) - A) - D["D_minus_x"]* (A - np.roll(A,-1,axis=0)) )
    SP_y = 1/(dy*dy) * (D["D_plus_y"]* (np.roll(A,1,axis=1) - A) - D["D_minus_y"]* (A - np.roll(A,-1,axis=1)) )
    SP_z = 1/(dz*dz) * (D["D_plus_z"]* (np.roll(A,1,axis=2) - A) - D["D_minus_z"]* (A - np.roll(A,-1,axis=2)) )
    SP = SP_x + SP_y + SP_z
    diff_A = (SP + f*np.multiply(A,1-A)) * dt
    A += diff_A
    return A

def gauss_sol3d(x,y,z):
    #experimentally chosen
    Dt = 15.0
    M = 1500
    
    gauss = M/np.power(4*np.pi * Dt,3/2) * np.exp(- (np.power(x,2) + np.power(y,2) + np.power(z,2))/(4*Dt))
    gauss = np.where(gauss>0.1, gauss,0)
    gauss = np.where(gauss>1, np.float64(1),gauss)
    return gauss


def get_initial_configuration(Nx,Ny,Nz,NxT,NyT,NzT,r):
    A =  np.zeros([Nx,Ny,Nz])
    if r == 0:
        A[NxT, NyT,NzT] = 1
    else:
        A[NxT-r:NxT+r, NyT-r:NyT+r,NzT-r:NzT+r] = 1
    
    return A

def solver(params):
    # Unpack parameters
    Dw = params['Dw']
    f = params['rho']
    RatioDw_Dg = params['RatioDw_Dg']
    sGM = params['gm']
    sWM = params['wm']
    NxT1_pct = params['NxT1_pct']
    NyT1_pct = params['NyT1_pct']
    NzT1_pct = params['NzT1_pct']
    
    assert isinstance(sGM, np.ndarray), "sGM must be a numpy array"
    assert isinstance(sWM, np.ndarray), "sWM must be a numpy array"
    assert sGM.ndim == 3, "sGM must be a 3D numpy array"
    assert sWM.ndim == 3, "sWM must be a 3D numpy array"
    assert sGM.shape == sWM.shape
    assert 0 <= NxT1_pct <= 1, "NxT1_pct must be between 0 and 1"
    assert 0 <= NyT1_pct <= 1, "NyT1_pct must be between 0 and 1"
    assert 0 <= NzT1_pct <= 1, "NzT1_pct must be between 0 and 1"

    # update in time
    days = 100
    # grid size
    Nx = sGM.shape[0]
    Ny = sGM.shape[1]
    Nz = sGM.shape[2]

    # grid steps
    dx =  1
    dy =  1
    dz =  1

    # Calculate the absolute positions based on percentages
    NxT1 = int(NxT1_pct * Nx)
    NyT1 = int(NyT1_pct * Ny)
    NzT1 = int(NzT1_pct * Nz)

    r = 1
    Nt = days * 10 * np.power((Dw/0.05),1)
    dt = days/Nt

    N_simulation_steps = int(np.ceil(Nt))

    yv, xv,zv = np.meshgrid(np.arange(0,sGM.shape[1]), np.arange(0,sGM.shape[0]),np.arange(0,sGM.shape[2]))
    A = np.array(gauss_sol3d(xv - NxT1 ,yv - NyT1,zv-NzT1))
    col_res = np.zeros([2, Nx, Ny, Nz])
    col_res[0] = copy.deepcopy(A)  # init
    # Simulation code
    D_domain = get_D(sWM, sGM, 0.1, Dw, RatioDw_Dg)
        
    result = {}
    try:
        for t in range(N_simulation_steps):
            A = FK_update(A, D_domain, f, dt, dx, dy, dz)

        col_res[1] = copy.deepcopy(A)  # final
        
        # Save results in the result dictionary
        result['initial_state'] = col_res[0]
        result['final_state'] = col_res[1]
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
