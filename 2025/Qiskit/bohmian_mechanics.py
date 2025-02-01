import numpy as np
import matplotlib.pyplot as plt

# Parameters
slit_distance = 1.0  
wavelength = 0.5  
screen_distance = 5.0  
num_particles = 500  

# Initial positions of particles (randomized)
particle_positions = np.random.uniform(-2, 2, num_particles)

# Compute Bohmian trajectories
trajectories = []
for x in particle_positions:
    trajectory = x + np.sin((x / slit_distance) * np.pi) * wavelength * screen_distance
    trajectories.append(trajectory)

# Plot results
plt.figure(figsize=(8, 6))
plt.hist(trajectories, bins=50, density=True, alpha=0.7, color='blue', label="Bohmian Trajectories")
plt.xlabel("Final Screen Position")
plt.ylabel("Probability Density")
plt.legend()
plt.title("Bohmian Interpretation of Double-Slit Experiment")
plt.show()
