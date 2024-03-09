import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
import torchvision.transforms as transforms
import os

# Define your CNN architecture
class WatermarkRemovalModel(nn.Module):
    def __init__(self):
        super(WatermarkRemovalModel, self).__init__()
        
        # Define the layers of your neural network
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
    def forward(self, x):
        # Define the forward pass of your neural network
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        return x

# Load your dataset
def load_dataset(data_dir):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.CenterCrop(224),      
        transforms.ToTensor(),           
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cleaned_dataset = ImageFolder(root=os.path.join(data_dir, 'cleaned'), transform=transform)
    uncleaned_dataset = ImageFolder(root=os.path.join(data_dir, 'uncleaned'), transform=transform)

    # Ensure both datasets are synchronized
    assert len(cleaned_dataset) == len(uncleaned_dataset), "Number of images in 'cleaned' and 'uncleaned' folders must be the same."

    # Concatenate both datasets
    dataset = ConcatDataset([cleaned_dataset, uncleaned_dataset])

    return dataset

def split_dataset(data_dir, batch_size=32, val_split=0.2, shuffle=True, num_workers=4):
    # Load the dataset
    dataset = ImageFolder(root=data_dir)

    # Calculate the split indices
    total_samples = len(dataset)
    indices = list(range(total_samples))
    split = int(total_samples * val_split)

    # Shuffle the indices if required
    if shuffle:
        torch.manual_seed(42)  # Set random seed for reproducibility
        torch.random.shuffle(indices)

    # Split the indices
    train_indices, val_indices = indices[split:], indices[:split]

    # Define samplers for the training and validation sets
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create data loaders for training and validation sets
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)

    return train_loader, val_loader


# Hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# # Load your dataset
# train_dataset = load_dataset('path/to/train/dataset')
# val_dataset = load_dataset('path/to/validation/dataset')

# # Create data loaders
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

train_loader, val_loader = split_dataset()

# Initialize your model
model = WatermarkRemovalModel()

# Define your loss function
criterion = nn.MSELoss()

# Define your optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Validation loop
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for images, targets in val_loader:
            outputs = model(images)
            val_loss += criterion(outputs, targets).item()
        
        val_loss /= len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), '/models/watermark_removal_model.pth')

