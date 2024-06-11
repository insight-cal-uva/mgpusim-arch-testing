import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np

# Step 1: Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)

# Step 2: Normalize the dataset to have values between 0 and 1
# This step is already handled by transforms.ToTensor() which converts the images to [0, 1] range.

# Step 3: Flatten the images
def flatten_and_convert_to_numpy(dataset):
    images = []
    labels = []
    for img, label in dataset:
        images.append(img.view(-1).numpy())
        labels.append(label)
    return np.array(images), np.array(labels)

train_images, train_labels = flatten_and_convert_to_numpy(mnist_train)
test_images, test_labels = flatten_and_convert_to_numpy(mnist_test)

# Step 4: Save the flattened data in a binary format
train_images.tofile('mnist_train_images.bin')
train_labels.tofile('mnist_train_labels.bin')
test_images.tofile('mnist_test_images.bin')
test_labels.tofile('mnist_test_labels.bin')

print("MNIST dataset saved as flattened binary format.")

