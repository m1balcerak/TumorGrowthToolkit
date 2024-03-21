#%%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 10          # Length of the domain
D = 1.0         # Diffusion coefficient
Nx = 100        # Number of spatial steps
Nt = 200        # Number of time steps
dx = L / Nx     # Spatial step size
dt = 0.001       # Time step size

# Stability condition for the diffusion equation
if D * dt / dx**2 > 0.5:
    raise ValueError("Stability condition failed: D*dt/dx^2 should be less than 0.5")

# Initialize u
x = np.linspace(0, L, Nx)
u = np.exp(-((x - L/2)**2) / 0.1)  # Initial condition: Gaussian distribution

# Matrix to store the solution at each time step
u_matrix = np.zeros((Nt, Nx))
u_matrix[0, :] = u

# Finite Difference Method
for n in range(1, Nt):
    for i in range(1, Nx-1):
        u_matrix[n, i] = u_matrix[n-1, i] +  2*(D * dt / dx**2) * (u_matrix[n-1, i+1] - 2*u_matrix[n-1, i] + u_matrix[n-1, i-1])

# Plotting
plt.figure(figsize=(10, 6))

# Select time steps to plot
time_steps = [0, 50, 100, 150, 199]
for t in time_steps:
    plt.plot(x, u_matrix[t, :], label=f't={t*dt:.2f}')

plt.title('Diffusion of a Gaussian distribution over time')
plt.xlabel('Position')
plt.ylabel('Concentration')
plt.legend()
plt.grid(True)
plt.show()
# %%
# Mean position (center of the domain)
x_mean = np.mean(x)

# Calculating the MSD from the simulation data
msd_from_data = np.array([np.sum((x - x_mean)**2 * u_matrix[t, :]) for t in range(Nt)])

# Plotting the MSD from the simulation data over time
plt.figure(figsize=(10, 6))
plt.plot( range(0, Nt), msd_from_data, label='MSD from Data')

plt.title('Mean Square Displacement over Time from Simulation Data')
plt.xlabel('Time')
plt.ylabel('MSD')
plt.grid(True)
plt.legend()
plt.show()
# %%
