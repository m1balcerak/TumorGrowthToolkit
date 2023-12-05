
# Tumor Growth Solver

This repository contains the `forwardFK_FDM` package, which includes a solver for simulating tumor growth using the Fisher-Kolmogorov model.

## Installation

To use the solver, first install the package by cloning this repository, going to the setup.py directory and using:
```
pip install .
```

## Usage

Here's a basic example of how to use the solver:

```python
from forwardFK_FDM.solver import solver
import numpy as np

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
    'NzT1_pct': 0.5
}

result = solver(parameters)

if result['success']:
    print("Simulation successful!")
    # Process the results here
else:
    print("Error occurred:", result['error'])
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
