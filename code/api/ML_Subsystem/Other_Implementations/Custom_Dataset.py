import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from skimage import io, transform

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



class CustomDatasetFromFile(Dataset):

    # folder_path (string): path to one species folder
    def __init__(self, folder_path):
        # Get image list
        self.image_list = glob.glob(folder_path + '*')

        # remove non .jpg
        for err in self.image_list:
            if err[-4:] != '.jpg':
                self.image_list.remove(err)

        # Calculate len
        self.data_len = len(self.image_list)

    def __getitem__(self, index):

        # Get image name from the pandas df
        single_image_path = self.image_list[index]

        # Open image
        im_as_im = Image.open(single_image_path)

        # Do some operations on image
        # Convert to numpy, dim = 28x28
        im_as_np = np.asarray(im_as_im)/255
        # Add channel dimension, dim = 1x28x28
        # Note: You do not need to do this if you are reading RGB images
        # or i there is already channel dimension
        im_as_np = np.expand_dims(im_as_np, 0)
        # Some preprocessing operations on numpy array
        # ...
        # ...
        # ...

        # Transform image to tensor, change data type
        im_as_ten = torch.from_numpy(im_as_np).float()

        # get label of the image
        label = single_image_path.split("/")[-2]
        return im_as_ten, label

    def __len__(self):
        return self.data_len
