import math
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms, models  # torchvision package contains many types of datasets (including MNIST dataset)
import numpy as np
import matplotlib.pyplot as plt


def image_convert_to_numpy(tensor):
    image = tensor.clone().detach().cpu().numpy()  # clones to tensor and transforms to numpy array. OR tensor.cpu().clone().detach().numpy()
    image = image.squeeze()
    image = image.transpose(1, 2, 0)
    # print(image.shape)                                                                            # (28, 28, 1.jpg)
    # denormalize image
    image = image * np.array((0.5,)) + np.array((0.5,))
    image = image.clip(0, 1)
    return image

def show_images(images):
    fig = plt.figure(figsize=(25, 4))

    grid = 20
    if len(images) < grid:
        grid = len(images)

    for index in np.arange(grid):
        ax = fig.add_subplot(2, math.ceil(grid/2), index + 1, xticks=[], yticks=[])
        plt.imshow(images[index])

    plt.axis('off')
    plt.show()


transform = transforms.Compose([transforms.Resize((300, 300)),  # resizes each image (pixels)
                                          transforms.RandomHorizontalFlip(),  # horizontal flip (lift to right)
                                          # random rotation hinders the performance
                                          transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Affine Type Transformations (stretch, scale)
                                          transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # changes color (this time, use 1.jpg)
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])

image_list = []
transformed_image_list = []

path_ = '/Users/omerunlusoy/Desktop/CS 492/PolliVidis-Neural-Pollen-Classification/datasets/Ankara_Dataset_cropped/alnus_glutinosa/'
for i, filename in enumerate(os.listdir(path_)):
    if i >= 20:
        break

    if filename.endswith(".jpg"):
        file = os.path.join(path_, filename)
        image_ = Image.open(file)
        image_list.append(image_)
        transformed_image = transform(image_)
        transformed_image = image_convert_to_numpy(transformed_image)
        transformed_image_list.append(transformed_image)

show_images(image_list)
show_images(transformed_image_list)

