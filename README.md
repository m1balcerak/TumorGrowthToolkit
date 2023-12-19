
# TumorGrowthToolkit

Welcome to TumorGrowthToolkit, a Python package dedicated to the simulation and analysis of tumor growth using numerical solvers for Partial Differential Equations. 

## Models
- Fisher-Kolmogorov FDM Solver (FisherKolmogorow)
- ...
  
## Installation

To use the solver, first install the package by cloning this repository, going to the setup.py directory and using:
```
pip install .
```

use without clone installation:
```
pip install git+https://github.com/m1balcerak/TumorGrowthToolkit.git
```

## Usage

Example usage:
```
python FK_example.py
```
Outputs a dictonary with results. Produces plots of the tumor progression in a random anathomy:
![Example Image](figures/plot2.png)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
