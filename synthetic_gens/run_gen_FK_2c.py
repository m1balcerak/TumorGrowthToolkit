import subprocess
import numpy as np
import os

def generate_random_parameters():
    # Replace these ranges with your desired ranges
    params = {
        'Dw': np.random.uniform(low=0.05, high=4),
        'rho': np.random.uniform(low=0.01, high=0.8),
        'lambda_np': np.random.uniform(0.05, 0.7),
        'sigma_np': np.random.uniform(0.05, 0.7),
        'D_s': np.random.uniform(0.05, 4),
        'lambda_s': np.random.uniform(0.01, 0.5),
        'NxT1_pct': np.random.uniform(low=0.3, high=0.7),
        'NyT1_pct': np.random.uniform(low=0.3, high=0.7),
        'NzT1_pct': np.random.uniform(low=0.3, high=0.7),
        'resolution_factor': 0.65,
        'RatioDw_Dg': 10,
        'gm_path': '../dataset/GM.nii.gz',  # Relative path to GM file
        'wm_path': '../dataset/WM.nii.gz',   # Relative path to WM file
        'th_necro_n': np.random.uniform(low=0.05, high=0.2),  # Threshold for necrotic core
        'th_enhancing_p': np.random.uniform(low=0.25, high=0.6),  # Threshold for enhancing core
        'th_edema_p': np.random.uniform(low=0.12, high=0.20)  # Threshold for edema
    }

    return params


def construct_command(base_output_dir, i, params):
    output_dir = f"{base_output_dir}_run{i}"
    os.makedirs(output_dir, exist_ok=True)

    command = ['python', 'generatePatient_FK_2c.py']
    for key, value in params.items():
        command.extend([f'--{key}', str(value)])
    command.extend(['--output_dir', output_dir])

    return command

def main():
    np.random.seed(42000)
    base_output_dir = "synthetic_runs1T_FK_2c/synthetic1T"
    num_iterations = 4

    processes = []

    for i in range(num_iterations):
        params = generate_random_parameters()
        command = construct_command(base_output_dir, i, params)
        command_str = ' '.join(command)
        print('Running:', command_str)

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append((i, process))

    for i, process in processes:
        stdout, stderr = process.communicate()  # Waits for the process to finish
        if process.returncode != 0:
            print(f"Error in run {i}:", stderr)
        else:
            print(f"Output from run {i}:", stdout)

if __name__ == "__main__":
    main()
