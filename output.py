import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Define both SimpleCNN and ComplexCNN models
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 25 * 25, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Training and evaluation functions
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def evaluate_model(model, structure, overall_filling_rate, num_slices_range):
    errors = []
    for num_slices in num_slices_range:
        slices = get_slices(structure, num_slices)
        dataset = ImageDataset(slices, overall_filling_rate)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        preds = []
        for inputs, _ in dataloader:
            with torch.no_grad():
                outputs = model(inputs).numpy()
            preds.extend(outputs)
        preds = np.array(preds).flatten()
        error = rmspe(np.array([overall_filling_rate] * len(preds)), preds)
        errors.append(error)
    return errors

# Define helper functions
def get_slices(structure, num_slices):
    return structure[:, :, :num_slices]

class ImageDataset(Dataset):
    def __init__(self, slices, overall_filling_rate):
        self.slices = slices
        self.overall_filling_rate = overall_filling_rate

    def __len__(self):
        return self.slices.shape[2]

    def __getitem__(self, idx):
        slice = self.slices[:, :, idx]
        slice = torch.tensor(slice, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(self.overall_filling_rate, dtype=torch.float32)
        return slice, label

# Load data and calculate overall filling rate
Structure = np.random.randint(0, 3, (100, 100, 100))  # Reduced size to avoid memory error

def calculate_filling_rate(structure):
    total_voxels = np.prod(structure.shape)
    material1_voxels = np.sum(structure == 1)
    material2_voxels = np.sum(structure == 2)
    filling_rate = (material1_voxels + material2_voxels) / total_voxels
    return filling_rate

overall_filling_rate = calculate_filling_rate(Structure)
print("Overall Filling Rate:", overall_filling_rate)

# Initialize models, criterion, and optimizers
simple_model = SimpleCNN()
complex_model = ComplexCNN()
criterion = nn.MSELoss()
simple_optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
complex_optimizer = optim.Adam(complex_model.parameters(), lr=0.001)

# Training the SimpleCNN model
slices = get_slices(Structure, 10)
dataset = ImageDataset(slices, overall_filling_rate)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
train_model(simple_model, dataloader, criterion, simple_optimizer, num_epochs=10)

# Training the ComplexCNN model
train_model(complex_model, dataloader, criterion, complex_optimizer, num_epochs=10)

# Evaluating the models
num_slices_range = range(1, 101, 5)  # Evaluate every 5 slices to reduce computation time
simple_errors = evaluate_model(simple_model, Structure, overall_filling_rate, num_slices_range)
complex_errors = evaluate_model(complex_model, Structure, overall_filling_rate, num_slices_range)

# Plotting RMSPE vs. Number of Slices for both models
plt.figure(figsize=(10, 6))
plt.plot(num_slices_range, simple_errors, marker='o', linestyle='-', color='b', label='SimpleCNN')
plt.plot(num_slices_range, complex_errors, marker='x', linestyle='--', color='r', label='ComplexCNN')
plt.xlabel('Number of Slices')
plt.ylabel('RMSPE')
plt.title('RMSPE vs. Number of Slices')
plt.legend()
plt.grid(True)
plt.savefig('/mnt/data/rmspe_comparison.png')  # Save the figure to current workspace
plt.show()
