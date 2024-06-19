import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Function to place a particle in the 2D structure
def place_particle_2d(structure, center, radius, value):
    y_center, x_center = center
    y, x = np.ogrid[:structure.shape[0], :structure.shape[1]]
    distance = (y - y_center)**2 + (x - x_center)**2
    mask = distance <= radius**2
    structure[mask] = value

# Function to check if a new particle overlaps with existing particles using KDTree
def check_overlap_2d(center, radius, particle_centers, particle_radii):
    if not particle_centers:
        return False
    tree = cKDTree(particle_centers)
    indices = tree.query_ball_point(center, radius + max(particle_radii))
    for idx in indices:
        existing_center = particle_centers[idx]
        existing_radius = particle_radii[idx]
        distance = np.linalg.norm(np.array(center) - np.array(existing_center))
        if distance < radius + existing_radius:
            return True
    return False

# Function to generate a single 2D figure
def generate_2d_figure():
    shape = (200, 200)
    structure = np.zeros(shape, dtype=np.uint8)

    total_area = shape[0] * shape[1]
    target_fill_area = total_area * np.random.uniform(0.7, 0.8)

    large_radius = 10
    small_radius = 5

    large_particle_area = np.pi * (large_radius ** 2)
    small_particle_area = np.pi * (small_radius ** 2)

    proportion_large = 0.3
    proportion_small = 0.7

    num_large_particles = int((target_fill_area * proportion_large) / large_particle_area)
    num_small_particles = int((target_fill_area * proportion_small) / small_particle_area)

    particle_centers = []
    particle_radii = []

    for _ in range(num_large_particles):
        while True:
            center = (np.random.randint(large_radius, shape[0] - large_radius),
                      np.random.randint(large_radius, shape[1] - large_radius))
            if not check_overlap_2d(center, large_radius, particle_centers, particle_radii):
                place_particle_2d(structure, center, large_radius, 2)
                particle_centers.append(center)
                particle_radii.append(large_radius)
                break

    for _ in range(num_small_particles):
        while True:
            center = (np.random.randint(small_radius, shape[0] - small_radius),
                      np.random.randint(small_radius, shape[1] - small_radius))
            if not check_overlap_2d(center, small_radius, particle_centers, particle_radii):
                place_particle_2d(structure, center, small_radius, 1)
                particle_centers.append(center)
                particle_radii.append(small_radius)
                break

    return structure

# Generate and visualize 20 different 2D figures
fig, axs = plt.subplots(4, 5, figsize=(20, 16))
for i in range(4):
    for j in range(5):
        structure = generate_2d_figure()
        axs[i, j].imshow(structure, cmap='viridis', interpolation='none')
        axs[i, j].set_title(f'Figure {i * 5 + j + 1}')
        axs[i, j].set_xlabel('X-axis')
        axs[i, j].set_ylabel('Y-axis')
        plt.colorbar(axs[i, j].imshow(structure, cmap='viridis', interpolation='none'), ax=axs[i, j], label='Value')

plt.tight_layout()
plt.show()
