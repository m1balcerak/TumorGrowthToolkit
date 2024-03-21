#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Grid and time parameters
Nx, Ny, Nz = 50, 50, 50  # Number of grid points in x, y, and z
Lx, Ly, Lz = 10.0, 10.0, 10.0  # Size of the grid in x, y, and z
dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz  # Spatial step sizes
dt = 0.01  # Time step

# Time duration
T = 1.0  # Total time
Nt = int(T / dt)  # Number of time steps

# Diffusion tensor
D = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])  # Diagonal tensor for simplicity

# Initialize concentration field
u = np.zeros((Nx, Ny, Nz))

# Initial condition (e.g., a peak in the center)
u[Nx // 2, Ny // 2, Nz // 2] = 1.0

# Function to compute the diffusion term using central differences
def diffusion_term(u, D, dx, dy, dz):
    dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dy)
    dudz = (np.roll(u, -1, axis=2) - np.roll(u, 1, axis=2)) / (2 * dz)
    
    d2udx2 = (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / dx**2
    d2udy2 = (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / dy**2
    d2udz2 = (np.roll(u, -1, axis=2) - 2 * u + np.roll(u, 1, axis=2)) / dz**2
    
    return D[0, 0] * d2udx2 + D[1, 1] * d2udy2 + D[2, 2] * d2udz2

# Time stepping loop
for t in range(Nt):
    u += dt * diffusion_term(u, D, dx, dy, dz)

# Plot the final state (slicing at the middle of z-axis for visualization)
plt.imshow(u[:, :, Nz // 2], extent=[0, Lx, 0, Ly], origin='lower')
plt.colorbar()
plt.title('Concentration at t = {:.2f}, z-slice'.format(T))
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %% 3D
import numpy as np
import matplotlib.pyplot as plt

# Grid and time parameters
Nx, Ny, Nz = 50, 50, 50  # Number of grid points in x, y, and z
Lx, Ly, Lz = 10.0, 10.0, 10.0  # Size of the grid in x, y, and z
dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz  # Spatial step sizes
dt = 0.01  # Time step

# Time duration
T = 10.0  # Total time
Nt = int(T / dt)  # Number of time steps

# Function to generate space-dependent diffusion tensor
def generate_diffusion_tensor(Nx, Ny, Nz, Lx, Ly, Lz):
    D = np.zeros((Nx, Ny, Nz, 3, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                x, y, z = i * dx, j * dy, k * dz
                # Example linear variation: D increases with x, y, z
                D[i, j, k] = np.array([[0.05 + 0.005 * x, 0, 0],
                                       [0, 0.05 + 0.005 * y, 0],
                                       [0, 0, 0.05 + 0.005 * z]])
    return D

# Initialize the space-dependent diffusion tensor
D = generate_diffusion_tensor(Nx, Ny, Nz, Lx, Ly, Lz)

# Initialize concentration field
u = np.zeros((Nx, Ny, Nz))
u[Nx // 2, Ny // 2, Nz // 2] = 1.0  # Initial condition (peak in the center)

# Function to compute the diffusion term using central differences
def diffusion_term(u, D, dx, dy, dz):
    # Compute spatial derivatives using central differences
    dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dy)
    dudz = (np.roll(u, -1, axis=2) - np.roll(u, 1, axis=2)) / (2 * dz)

    # Apply the diffusion tensor to the spatial derivatives
    dux = D[:, :, :, 0, 0] * dudx
    duy = D[:, :, :, 1, 1] * dudy
    duz = D[:, :, :, 2, 2] * dudz

    # Sum the contributions from each direction
    return dux + duy + duz


import numpy as np

def diffusion_termWithCrossDiffusion(u, D, dx, dy, dz):
    # Compute spatial derivatives
    dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
    dudy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dy)
    dudz = (np.roll(u, -1, axis=2) - np.roll(u, 1, axis=2)) / (2 * dz)

    # Apply the full diffusion tensor to the gradient
    Du = np.zeros_like(u)
    for i in range(3):
        Du += (D[:, :, :, i, 0] * dudx +
               D[:, :, :, i, 1] * dudy +
               D[:, :, :, i, 2] * dudz)

    # Compute divergence of Du
    divDu = (np.roll(Du, -1, axis=0) - np.roll(Du, 1, axis=0)) / (2 * dx) + \
            (np.roll(Du, -1, axis=1) - np.roll(Du, 1, axis=1)) / (2 * dy) + \
            (np.roll(Du, -1, axis=2) - np.roll(Du, 1, axis=2)) / (2 * dz)

    return divDu

# The rest of the code for setting up the problem remains the same


# The rest of the setup remains the same

# The rest of the code remains the same as in the previous example


# Time stepping loop
for t in range(Nt):
    u += dt * diffusion_term(u, D, dx, dy, dz)

# Plot the final state (slicing at the middle of z-axis for visualization)
plt.imshow(u[:, :, Nz // 2], extent=[0, Lx, 0, Ly], origin='lower')
plt.colorbar()
plt.title('Concentration at t = {:.2f}, z-slice'.format(T))
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# %%
