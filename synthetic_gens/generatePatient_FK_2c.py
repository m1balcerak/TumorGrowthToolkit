import argparse
import os
import nibabel as nib
import numpy as np
import json
import scipy.ndimage

from TumorGrowthToolkit.FK_2c import Solver

def gibbs_sampler_3d(shape, num_iter, noise_std_dev):
    field = np.random.rand(*shape)
    size_x, size_y, size_z = shape

    for it in range(num_iter):
        for i in range(size_x):
            for j in range(size_y):
                for k in range(size_z):
                    local_sum = np.sum(field[max(0, i-1):min(size_x, i+2), 
                                             max(0, j-1):min(size_y, j+2),
                                             max(0, k-1):min(size_z, k+2)])
                    local_count = (min(size_x, i+2) - max(0, i-1)) * (min(size_y, j+2) - max(0, j-1)) * (min(size_z, k+2) - max(0, k-1))
                    local_mean = local_sum / local_count
                    field[i, j, k] = np.random.normal(local_mean, noise_std_dev)

    return field

def process_fet_data(P_data, N_data, wm_data, gm_data, th_matter, necrosis_threshold=0.4):
    # Generate noise and add to P_data
    noise = gibbs_sampler_3d(shape=P_data.shape, num_iter=10, noise_std_dev=0.1)
    P_noisy = P_data + noise/3

    # Downsample and then Upsample back to original shape
    downsample_factor = 0.25
    upsample_factor = 1 / downsample_factor  # Upsample back to original size
    downsampled = scipy.ndimage.zoom(P_noisy, zoom=downsample_factor, order=1)
    upsampled = scipy.ndimage.zoom(downsampled, zoom=upsample_factor, order=1)

    # Ensure upsampled array matches original shape
    upsampled = scipy.ndimage.zoom(upsampled, zoom=np.array(P_data.shape) / np.array(upsampled.shape), order=1)

    # Brain mask
    brain_mask = (wm_data + gm_data) > th_matter

    # Add random noise within brain mask
    #random_noise = np.random.normal(loc=0, scale=0.001, size=P_data.shape)
    #fet_data = np.where(brain_mask, upsampled + random_noise, upsampled)
    fet_data = np.where(brain_mask, upsampled, 0)

    # Reduce FET signal where necrotic cells are above threshold
    #necrosis_mask = N_data > necrosis_threshold
    #fet_data[necrosis_mask] = 0  # or a very low value to simulate reduced signal

    return fet_data

def load_nifti_to_numpy(filepath):
    img = nib.load(filepath)
    return img.get_fdata()

def save_nifti(array, filename):
    img = nib.Nifti1Image(array, np.eye(4))
    nib.save(img, filename)

def create_segmentation_map(P, N, th_necro_n, th_enhancing_p, th_edema_p):
    # Initialize segmentation map
    segmentation_map = np.zeros(P.shape, dtype=int)

    # Segment edema
    edema_mask = (P >= th_edema_p) & (P < th_enhancing_p)
    segmentation_map[edema_mask] = 3

    # Segment enhancing core
    enhancing_core_mask = P >= th_enhancing_p
    segmentation_map[enhancing_core_mask] = 1

    # Segment necrotic core
    necrotic_core_mask = N > th_necro_n
    segmentation_map[necrotic_core_mask] = 4

    return segmentation_map

def save_parameters_as_json(params, filename):
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)

def save_parameters_as_text(params, filename):
    # Exclude gm_data and wm_data, use their file paths instead
    with open(filename, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
                


def main(args):
    
    # Save parameters as JSON and text file
    args_dict = vars(args)
    save_parameters_as_json(args_dict, os.path.join(args.output_dir, 'parameters.json'))
    save_parameters_as_text(args_dict, os.path.join(args.output_dir, 'parameters.txt'))
    # Load gm_data and wm_data from NIFTI files
    gm_data = load_nifti_to_numpy(args.gm_path)
    wm_data = load_nifti_to_numpy(args.wm_path)
    

    # Set up parameters based on command line arguments
    parameters = {
        'Dw': args.Dw,
        'rho': args.rho,
        'lambda_np': args.lambda_np,
        'sigma_np': args.sigma_np,
        'D_s': args.D_s,
        'lambda_s': args.lambda_s,
        'gm': gm_data,
        'wm': wm_data,
        'NxT1_pct': args.NxT1_pct,
        'NyT1_pct': args.NyT1_pct,
        'NzT1_pct': args.NzT1_pct,
        'resolution_factor': args.resolution_factor,
        'RatioDw_Dg': args.RatioDw_Dg,
        'th_matter': args.th_matter,
        'th_necro': args.th_necro,
        'dx_mm': args.dx_mm,
        'dy_mm': args.dy_mm,
        'dz_mm': args.dz_mm,
        'init_scale': args.init_scale,
        'time_series_solution_Nt': args.time_series_solution_Nt,
        'Nt_multiplier': args.Nt_multiplier
    }

    # Run the FK_solver
    fk_solver = Solver(parameters)
    result = fk_solver.solve()

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Save final states as NIFTI files
    final_states = result['final_state']
    save_nifti(final_states['P'], os.path.join(args.output_dir, 'P.nii.gz'))
    save_nifti(final_states['N'], os.path.join(args.output_dir, 'N.nii.gz'))
    save_nifti(final_states['S'], os.path.join(args.output_dir, 'S.nii.gz'))

    # Save wm_data and gm_data as NIFTI files
    save_nifti(wm_data, os.path.join(args.output_dir, 'wm_data.nii.gz'))
    save_nifti(gm_data, os.path.join(args.output_dir, 'gm_data.nii.gz'))

    # Save FET and segm
    fet_data = process_fet_data(final_states['P'], final_states['N'], wm_data, gm_data, args.th_matter)
    save_nifti(fet_data, os.path.join(args.output_dir, 'FET.nii.gz'))
    segmentation_map = create_segmentation_map(final_states['P'], final_states['N'], args.th_necro_n, args.th_enhancing_p, args.th_edema_p)
    save_nifti(segmentation_map, os.path.join(args.output_dir, 'segm.nii.gz'))
    
    print(f"Patient data saved in {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Synthetic Patient Data')
    parser.add_argument('--th_necro_n', type=float, required=True, help='Threshold for necrotic core')
    parser.add_argument('--th_enhancing_p', type=float, required=True, help='Threshold for enhancing core')
    parser.add_argument('--th_edema_p', type=float, required=True, help='Threshold for edema')
    parser.add_argument('--Dw', type=float, required=True)
    parser.add_argument('--rho', type=float, required=True)
    parser.add_argument('--lambda_np', type=float, required=True)
    parser.add_argument('--sigma_np', type=float, required=True)
    parser.add_argument('--D_s', type=float, required=True)
    parser.add_argument('--lambda_s', type=float, required=True)
    parser.add_argument('--gm_path', type=str, required=True, help='Path to the gm NIFTI file')
    parser.add_argument('--wm_path', type=str, required=True, help='Path to the wm NIFTI file')
    parser.add_argument('--NxT1_pct', type=float, required=True)
    parser.add_argument('--NyT1_pct', type=float, required=True)
    parser.add_argument('--NzT1_pct', type=float, required=True)
    parser.add_argument('--resolution_factor', type=float, required=True)
    parser.add_argument('--RatioDw_Dg', type=float, default=10.0)
    parser.add_argument('--th_matter', type=float, default=0.1)
    parser.add_argument('--th_necro', type=float, default=0.9)
    parser.add_argument('--dx_mm', type=float, default=1.0)
    parser.add_argument('--dy_mm', type=float, default=1.0)
    parser.add_argument('--dz_mm', type=float, default=1.0)
    parser.add_argument('--init_scale', type=float, default=1.0)
    parser.add_argument('--time_series_solution_Nt', type=int)
    parser.add_argument('--Nt_multiplier', type=int, default=10)
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save patient data')
    
    args = parser.parse_args()
    main(args)
