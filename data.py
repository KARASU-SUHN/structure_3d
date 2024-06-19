import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Helper functions
def place_particle_2d(structure, center, radius, value):
    y_center, x_center = center
    y, x = np.ogrid[:structure.shape[0], :structure.shape[1]]
    distance = (y - y_center)**2 + (x - x_center)**2
    mask = distance <= radius**2
    structure[mask] = value

def check_overlap_2d(center, radius, particle_centers, particle_radii):
    for existing_center, existing_radius in particle_centers:
        distance = np.linalg.norm(np.array(center) - np.array(existing_center))
        if distance < radius + existing_radius:
            return True
    return False

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
    return structure, target_fill_area / total_area  # return the structure and the fill rate

class SyntheticDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.data = []
        self.targets = []
        for _ in range(num_samples):
            structure, fill_rate = generate_2d_figure()
            self.data.append(structure)
            self.targets.append(fill_rate)
        
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, target

# Generate the dataset
num_samples = 1000
dataset = SyntheticDataset(num_samples)

# Split the dataset into training and test sets
train_size = int(0.8 * num_samples)
test_size = num_samples - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create DataLoader for training and test sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


import torch.nn as nn
import torch.nn.functional as F

class CNNRegression(nn.Module):
    def __init__(self):
        super(CNNRegression, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 25 * 25, 1000)
        self.fc2 = nn.Linear(1000, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Move the model to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNNRegression().to(device)


import torch.optim as optim

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")




# Function to calculate RMSPE
def rmspe(y_true, y_pred):
    return torch.sqrt(torch.mean(((y_true - y_pred) / y_true) ** 2))

# Evaluation loop
model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        y_true.extend(targets.cpu().numpy())
        y_pred.extend(outputs.cpu().numpy().flatten())

y_true = torch.tensor(y_true)
y_pred = torch.tensor(y_pred)
error = rmspe(y_true, y_pred)
print(f"RMSPE: {error.item()}")

