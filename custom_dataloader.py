import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import gzip
import os

class FashionMNISTCustomDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images, self.labels = self.load_data(images_path, labels_path)
        self.transform = transform
    
    def load_data(self, images_path, labels_path):
        with gzip.open(images_path, 'rb') as img_file:
            images = np.frombuffer(img_file.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
        with gzip.open(labels_path, 'rb') as lbl_file:
            labels = np.frombuffer(lbl_file.read(), dtype=np.uint8, offset=8)
        return images, labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0  # Normalize
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return torch.tensor(image, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)

# Paths to dataset (update if needed)
data_path = "./data"
train_images = os.path.join(data_path, "train-images-idx3-ubyte.gz")
train_labels = os.path.join(data_path, "train-labels-idx1-ubyte.gz")
test_images = os.path.join(data_path, "t10k-images-idx3-ubyte.gz")
test_labels = os.path.join(data_path, "t10k-labels-idx1-ubyte.gz")

# Create dataset and dataloader
train_dataset = FashionMNISTCustomDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = FashionMNISTCustomDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Custom DataLoader is ready!")
