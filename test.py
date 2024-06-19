# import numpy as np
# import matplotlib.pyplot as plt

# # Initialize the 3D structure
# shape = (200, 200, 200)
# structure = np.zeros(shape, dtype=np.uint8)

# # Function to place a particle in the 3D structure
# def place_particle(structure, center, radius, value):
#     z_center, y_center, x_center = center
#     z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
#     distance = (z - z_center)**2 + (y - y_center)**2 + (x - x_center)**2
#     mask = distance <= radius**2
#     structure[mask] = value

# # Randomly place large particles (value 2)
# num_large_particles = 10
# large_radius = 10
# for _ in range(num_large_particles):
#     center = (np.random.randint(large_radius, shape[0] - large_radius),
#               np.random.randint(large_radius, shape[1] - large_radius),
#               np.random.randint(large_radius, shape[2] - large_radius))
#     place_particle(structure, center, large_radius, 2)

# # Randomly place small particles (value 1)
# num_small_particles = 50
# small_radius = 5
# for _ in range(num_small_particles):
#     center = (np.random.randint(small_radius, shape[0] - small_radius),
#               np.random.randint(small_radius, shape[1] - small_radius),
#               np.random.randint(small_radius, shape[2] - small_radius))
#     place_particle(structure, center, small_radius, 1)

# # Visualize multiple slices of the structure
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# slices = [0, 100, 199]
# for i, slice_idx in enumerate(slices):
#     slice_data = structure[slice_idx, :, :]
#     axs[i].imshow(slice_data, cmap='viridis', interpolation='none')
#     axs[i].set_title(f'Heatmap of structure[{slice_idx}][*][*]')
#     axs[i].set_xlabel('X-axis')
#     axs[i].set_ylabel('Y-axis')
#     axs[i].colorbar = plt.colorbar(axs[i].imshow(slice_data, cmap='viridis', interpolation='none'), ax=axs[i], label='Value')

# plt.show()




# import numpy as np
# import matplotlib.pyplot as plt

# # Initialize the 3D structure
# shape = (200, 200, 200)
# structure = np.zeros(shape, dtype=np.uint8)

# # Function to place a particle in the 3D structure
# def place_particle(structure, center, radius, value):
#     z_center, y_center, x_center = center
#     z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
#     distance = (z - z_center)**2 + (y - y_center)**2 + (x - x_center)**2
#     mask = distance <= radius**2
#     structure[mask] = value

# # Randomly place more large particles (value 2)
# num_large_particles = 50  # Increased number of large particles
# large_radius = 10
# for _ in range(num_large_particles):
#     center = (np.random.randint(large_radius, shape[0] - large_radius),
#               np.random.randint(large_radius, shape[1] - large_radius),
#               np.random.randint(large_radius, shape[2] - large_radius))
#     place_particle(structure, center, large_radius, 2)

# # Randomly place more small particles (value 1)
# num_small_particles = 200  # Increased number of small particles
# small_radius = 5
# for _ in range(num_small_particles):
#     center = (np.random.randint(small_radius, shape[0] - small_radius),
#               np.random.randint(small_radius, shape[1] - small_radius),
#               np.random.randint(small_radius, shape[2] - small_radius))
#     place_particle(structure, center, small_radius, 1)

# # Visualize multiple slices of the structure
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# slices = [0, 100, 199]
# for i, slice_idx in enumerate(slices):
#     slice_data = structure[slice_idx, :, :]
#     cax = axs[i].imshow(slice_data, cmap='viridis', interpolation='none')
#     axs[i].set_title(f'Heatmap of structure[{slice_idx}][*][*]')
#     axs[i].set_xlabel('X-axis')
#     axs[i].set_ylabel('Y-axis')
#     plt.colorbar(cax, ax=axs[i], label='Value')

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # Initialize the 3D structure
# shape = (200, 200, 200)
# structure = np.zeros(shape, dtype=np.uint8)

# print(structure.shape)

# # Function to place a particle in the 3D structure
# def place_particle(structure, center, radius, value):
#     z_center, y_center, x_center = center
#     z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
#     distance = (z - z_center)**2 + (y - y_center)**2 + (x - x_center)**2
#     mask = distance <= radius**2
#     structure[mask] = value

# # Calculate the volume of the structure and the target fill volume
# total_volume = shape[0] * shape[1] * shape[2]
# target_fill_volume = total_volume * 0.7

# # Volume of a sphere formula: V = 4/3 * π * r^3
# def sphere_volume(radius):
#     return (4/3) * np.pi * (radius ** 3)

# # Define the radii of large and small particles
# large_radius = 10
# small_radius = 5

# # Calculate the number of large and small particles needed to achieve the target fill volume
# large_particle_volume = sphere_volume(large_radius)
# small_particle_volume = sphere_volume(small_radius)

# # Proportions for large and small particles (can be adjusted as needed)
# proportion_large = 0.3
# proportion_small = 0.7

# # Number of particles
# num_large_particles = int((target_fill_volume * proportion_large) / large_particle_volume)
# num_small_particles = int((target_fill_volume * proportion_small) / small_particle_volume)

# # Place large particles
# for _ in range(num_large_particles):
#     center = (np.random.randint(large_radius, shape[0] - large_radius),
#               np.random.randint(large_radius, shape[1] - large_radius),
#               np.random.randint(large_radius, shape[2] - large_radius))
#     place_particle(structure, center, large_radius, 2)

# # Place small particles
# for _ in range(num_small_particles):
#     center = (np.random.randint(small_radius, shape[0] - small_radius),
#               np.random.randint(small_radius, shape[1] - small_radius),
#               np.random.randint(small_radius, shape[2] - small_radius))
#     place_particle(structure, center, small_radius, 1)

# # Visualize multiple slices of the structure
# fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# slices = [0, 100, 199]
# for i, slice_idx in enumerate(slices):
#     slice_data = structure[slice_idx, :, :]
#     cax = axs[i].imshow(slice_data, cmap='viridis', interpolation='none')
#     axs[i].set_title(f'Heatmap of structure[{slice_idx}][*][*]')
#     axs[i].set_xlabel('X-axis')
#     axs[i].set_ylabel('Y-axis')
#     plt.colorbar(cax, ax=axs[i], label='Value')

# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Initialize the 3D structure
shape = (200, 200, 200)
structure = np.zeros(shape, dtype=np.uint8)

# Function to place a particle in the 3D structure
def place_particle(structure, center, radius, value):
    z_center, y_center, x_center = center
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    distance = (z - z_center)**2 + (y - y_center)**2 + (x - x_center)**2
    mask = distance <= radius**2
    structure[mask] = value

# Calculate the volume of the structure and the target fill volume
total_volume = shape[0] * shape[1] * shape[2]
target_fill_volume = total_volume * 0.7

# Volume of a sphere formula: V = 4/3 * π * r^3
def sphere_volume(radius):
    return (4/3) * np.pi * (radius ** 3)

# Define the radii of large and small particles
large_radius = 10
small_radius = 5

# Calculate the number of large and small particles needed to achieve the target fill volume
large_particle_volume = sphere_volume(large_radius)
small_particle_volume = sphere_volume(small_radius)

# Proportions for large and small particles (can be adjusted as needed)
proportion_large = 0.3
proportion_small = 0.7

# Number of particles
num_large_particles = int((target_fill_volume * proportion_large) / large_particle_volume)
num_small_particles = int((target_fill_volume * proportion_small) / small_particle_volume)

# Function to check if a new particle overlaps with existing particles
def check_overlap(center, radius, particle_centers):
    for existing_center, existing_radius in particle_centers:
        distance = np.linalg.norm(np.array(center) - np.array(existing_center))
        if distance < radius + existing_radius:
            return True
    return False

# Place particles without overlap
particle_centers = []

# Place large particles
for _ in range(num_large_particles):
    while True:
        center = (np.random.randint(large_radius, shape[0] - large_radius),
                  np.random.randint(large_radius, shape[1] - large_radius),
                  np.random.randint(large_radius, shape[2] - large_radius))
        if not check_overlap(center, large_radius, particle_centers):
            place_particle(structure, center, large_radius, 2)
            particle_centers.append((center, large_radius))
            break

# Place small particles
for _ in range(num_small_particles):
    while True:
        center = (np.random.randint(small_radius, shape[0] - small_radius),
                  np.random.randint(small_radius, shape[1] - small_radius),
                  np.random.randint(small_radius, shape[2] - small_radius))
        if not check_overlap(center, small_radius, particle_centers):
            place_particle(structure, center, small_radius, 1)
            particle_centers.append((center, small_radius))
            break

# Visualize multiple slices of the structure
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
slices = [0, 100, 199]
for i, slice_idx in enumerate(slices):
    slice_data = structure[slice_idx, :, :]
    cax = axs[i].imshow(slice_data, cmap='viridis', interpolation='none')
    axs[i].set_title(f'Heatmap of structure[{slice_idx}][*][*]')
    axs[i].set_xlabel('X-axis')
    axs[i].set_ylabel('Y-axis')
    plt.colorbar(cax, ax=axs[i], label='Value')

plt.show()
