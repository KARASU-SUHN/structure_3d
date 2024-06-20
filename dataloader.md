To create a dataset for a PyTorch regression problem where your target is a float number (the percentage of particles filling a square), you can use the `torch.utils.data.Dataset` class. Here's a step-by-step guide to load your data:

1. **Prepare your data**: You should have your figures (e.g., images) and their corresponding float numbers (fill rates) in a structured format.

2. **Create a custom Dataset class**: Inherit from `torch.utils.data.Dataset` and implement the `__init__`, `__len__`, and `__getitem__` methods.

3. **Create DataLoader**: Use `torch.utils.data.DataLoader` to create a data loader for batching and shuffling the data.

Here’s an example of how to do this:

### Step 1: Prepare your data
Assume you have your figures as images in a directory and their corresponding fill rates in a list.

```python
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# Example data
images_dir = 'path/to/images'
fill_rates = [78.688, 65.321, 89.456, ...]  # List of fill rates
image_filenames = ['image1.png', 'image2.png', 'image3.png', ...]  # Corresponding image filenames
```

### Step 2: Create a custom Dataset class
```python
class FillRateDataset(Dataset):
    def __init__(self, images_dir, image_filenames, fill_rates, transform=None):
        self.images_dir = images_dir
        self.image_filenames = image_filenames
        self.fill_rates = fill_rates
        self.transform = transform
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert('RGB')  # Assuming images are in RGB format
        fill_rate = self.fill_rates[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # Convert fill_rate to tensor
        fill_rate = torch.tensor(fill_rate, dtype=torch.float32)
        
        return image, fill_rate
```

### Step 3: Create DataLoader
```python
from torchvision import transforms

# Define any image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to a common size
    transforms.ToTensor()           # Convert images to PyTorch tensors
])

# Create dataset
dataset = FillRateDataset(images_dir, image_filenames, fill_rates, transform=transform)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example of using the dataloader
for images, fill_rates in dataloader:
    print(images.shape)  # Shape of image batch
    print(fill_rates.shape)  # Shape of fill rate batch
    # Your training code here
```

### Summary:
1. **Dataset Preparation**: Ensure your images and fill rates are organized.
2. **Custom Dataset Class**: Implement the custom dataset class.
3. **DataLoader**: Use DataLoader for batching and shuffling.

This setup will help you load your data efficiently for training a regression model in PyTorch.