
# Tumor Growth Solver

This repository contains the `forwardFK_FDM` package, which includes a solver for simulating tumor growth using the Fisher-Kolmogorov model.

## Installation

To use the solver, first install the package by cloning this repository, going to the setup.py directory and using:
```
pip install .
```

use without clone installation:
```
pip install git+https://github.com/m1balcerak/forwardFK_FDM.git
```

## Usage

Here's a basic example of how to use the solver:

```python
from forwardFK_FDM.solver import solver
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

# Example grey matter and white matter data
# Generate random data for grey matter
gm_data = np.random.rand(100, 100, 100)

# Calculate white matter data as the complementary probability
wm_data = 1 - gm_data

parameters = {
    'Dw': 0.05,         # Diffusion coefficient for white matter
    'rho': 0.2,        # Proliferation rate
    'RatioDw_Dg': 1.5,  # Ratio of diffusion coefficients in white and grey matter
    'gm': gm_data,      # Grey matter data
    'wm': wm_data,      # White matter data
    'NxT1_pct': 0.5,    # initial focal position (in percentages)
    'NyT1_pct': 0.5,
    'NzT1_pct': 0.5,
    'resolution_factor': 1
}

# Step 2: Create custom color maps
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['black', 'white'], 256)
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap2', ['black', 'green', 'yellow', 'red'], 256)

# Step 3: Calculate the slice index
NzT = int(parameters['NzT1_pct'] * gm_data.shape[2])

# Step 4: Plotting function
def plot_tumor_states(wm_data, initial_state, final_state, slice_index):
    plt.figure(figsize=(12, 6))

    # Plot initial state
    plt.subplot(1, 2, 1)
    plt.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
    plt.imshow(initial_state[:, :, slice_index], cmap=cmap2, vmin=0, vmax=1, alpha=0.65)
    plt.title("Initial Tumor State")

    # Plot final state
    plt.subplot(1, 2, 2)
    plt.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
    plt.imshow(final_state[:, :, slice_index], cmap=cmap2, vmin=0, vmax=1, alpha=0.65)
    plt.title("Final Tumor State")

    plt.show()

# Step 5: Integrate into your existing code
result = solver(parameters)
if result['success']:
    print("Simulation successful!")
    plot_tumor_states(wm_data, result['initial_state'], result['final_state'], NzT)
else:
    print("Error occurred:", result['error'])
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
