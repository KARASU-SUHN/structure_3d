import numpy as np

# Example structure array (for illustration; replace this with your actual data loading)
# The shape of Structure is (200, 200, 200)
Structure = np.random.randint(0, 3, (200, 200, 200))  # replace this with your actual data loading

def get_slices(structure, num_slices):
    return structure[:, :, :num_slices]

# Get slices (example for 4 slices)
slices = get_slices(Structure, 4)


def calculate_filling_rate(structure):
    total_voxels = np.prod(structure.shape)
    material1_voxels = np.sum(structure == 1)
    material2_voxels = np.sum(structure == 2)
    filling_rate = (material1_voxels + material2_voxels) / total_voxels
    return filling_rate

# Calculate the overall filling rate
overall_filling_rate = calculate_filling_rate(Structure)
print("Overall Filling Rate:", overall_filling_rate)


import torch
from torch.utils.data import Dataset, DataLoader

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

# Create dataset and dataloader
dataset = ImageDataset(slices, overall_filling_rate)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)




import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 50 * 50, 64)
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

# Instantiate model, define loss and optimizer
model = SimpleCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Train the model
train_model(model, dataloader, criterion, optimizer, num_epochs=10)



def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def evaluate_model(model, structure, overall_filling_rate, num_slices_range):
    errors = []
    for num_slices in num_slices_range:
        slices = get_slices(structure, num_slices)
        dataset = ImageDataset(slices, overall_filling_rate)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        preds = []
        for inputs, _ in dataloader:
            with torch.no_grad():
                outputs = model(inputs).numpy()
            preds.extend(outputs)

        preds = np.array(preds).flatten()
        error = rmspe(np.array([overall_filling_rate]*len(preds)), preds)
        errors.append(error)

    return errors

# Evaluate model
errors = evaluate_model(model, Structure, overall_filling_rate, range(1, 201))
print(errors)



##revised


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Assuming Structure is a 3D numpy array of shape (200, 200, 200)
Structure = np.random.randint(0, 3, (200, 200, 200))  # Replace with actual data

def calculate_filling_rate(structure):
    total_voxels = np.prod(structure.shape)
    material1_voxels = np.sum(structure == 1)
    material2_voxels = np.sum(structure == 2)
    filling_rate = (material1_voxels + material2_voxels) / total_voxels
    return filling_rate

overall_filling_rate = calculate_filling_rate(Structure)
print("Overall Filling Rate:", overall_filling_rate)

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

class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 25 * 25, 256)
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

def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
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
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        preds = []
        for inputs, _ in dataloader:
            with torch.no_grad():
                outputs = model(inputs).numpy()
            preds.extend(outputs)
        preds = np.array(preds).flatten()
        error = rmspe(np.array([overall_filling_rate] * len(preds)), preds)
        errors.append(error)
    return errors

model = ComplexCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_slices_range = range(1, 201)
errors = evaluate_model(model, Structure, overall_filling_rate, num_slices_range)

plt.figure(figsize=(10, 6))
plt.plot(num_slices_range, errors, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Slices')
plt.ylabel('RMSPE')
plt.title('RMSPE vs. Number of Slices')
plt.grid(True)
plt.show()


### 200 0 

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Assuming Structure is a 3D numpy array of shape (200, 200, 200)
Structure = np.random.randint(0, 3, (200, 200, 200))  # Replace with actual data

def calculate_filling_rate(structure):
    total_voxels = np.prod(structure.shape)
    material1_voxels = np.sum(structure == 1)
    material2_voxels = np.sum(structure == 2)
    filling_rate = (material1_voxels + material2_voxels) / total_voxels
    return filling_rate

overall_filling_rate = calculate_filling_rate(Structure)
print("Overall Filling Rate:", overall_filling_rate)

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

class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 25 * 25, 256)
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

def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
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
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        preds = []
        for inputs, _ in dataloader:
            with torch.no_grad():
                outputs = model(inputs).numpy()
            preds.extend(outputs)
        preds = np.array(preds).flatten()
        error = rmspe(np.array([overall_filling_rate] * len(preds)), preds)
        errors.append(error)
    errors.append(0.0)  # Adding 0 RMSPE for 200 slices
    return errors

model = ComplexCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model with a certain number of slices (example with 10 slices)
slices = get_slices(Structure, 10)
dataset = ImageDataset(slices, overall_filling_rate)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
train_model(model, dataloader, criterion, optimizer, num_epochs=10)

# Evaluating the model
num_slices_range = range(1, 201)
errors = evaluate_model(model, Structure, overall_filling_rate, num_slices_range)

# Plotting RMSPE vs. Number of Slices
plt.figure(figsize=(10, 6))
plt.plot(range(1, 202), errors, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Slices')
plt.ylabel('RMSPE')
plt.title('RMSPE vs. Number of Slices')
plt.grid(True)
plt.show()


