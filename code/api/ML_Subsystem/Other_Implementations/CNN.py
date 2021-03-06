from datetime import timedelta

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models  # torchvision package contains many types of datasets (including MNIST dataset)
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time

from Helper_Functions import Helper_Functions

warnings.filterwarnings("ignore")  # suppress all warnings

########################################################################################################################

""" CNN Class """
class CNN(nn.Module):

    def __init__(self, classes, image_size):
        super().__init__()

        # self parameters
        self.classes = classes

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # specifies run device for more optimum runtime

        self.image_size = image_size
        self.freeze_AlexNet_until_layer = 7

        # self model
        self.model = models.AlexNet(num_classes=len(self.classes))

        # freeze some layers with setting requires_grad to False
        for i in range(self.freeze_AlexNet_until_layer):
            for param in self.model.features[i].parameters():
                param.requires_grad = False

        self.model.to(device=self.device)

    def forward(self, X):
        return self.model.forward(X)

    def forward_image(self, img, transform_validation):
        img = transform_validation(img)
        img = img.unsqueeze(0)
        output = self.model.forward(img)
        _, predicted_classes = torch.max(output, 1)  # gets the maximum output value for each output
        return self.classes[predicted_classes.item()]

    def load_model(self):
        print('! CNN.load_Model()')
        return torch.load('models/best_model.tf')


########################################################################################################################
# hyper parameters
learning_rate = 0.001
epochs = 50
batch_size = 32

# freeze_AlexNet_until_layer = 7

# dataset parameters
image_size = 200
train_validation_split_ratio = 0.8
dataset_path = '/Users/omerunlusoy/Desktop/CS 492/PolliVidis-Neural-Pollen-Classification/datasets/Ankara_Dataset_cropped'

# print variables
print_initial_dataset = False
plot_loss_and_corrects = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # specifies run device for more optimum runtime

########################################################################################################################

def train(model, training_loader, validation_loader, criterion, optimizer, helper_functions):
    # iterations
    losses = []
    corrects = []
    validation_losses = []
    validation_corrects = []

    print('Training begins...')
    start_training_time = time.time()
    for e in range(epochs):
        running_loss = 0.0
        running_corrects = 0.0
        validation_running_loss = 0.0
        validation_running_corrects = 0.0

        for images, labels in training_loader:  # for each epoch, iterate through each training batch (size of bach_size)

            images = images.to(device)
            labels = labels.to(device)

            outputs = model.forward(images)  # make a prediction (outputs of NN), (y_pred)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()  # zeros (resets grad), (grads accumulates after each backward)
            loss.backward()  # derivative (gradient) of loss
            optimizer.step()  # optimizer recalculates params (can be called after loss.backward())

            _, predicted_classes = torch.max(outputs, 1)  # gets the maximum output value for each output
            num_correct_predictions = torch.sum(predicted_classes == labels.data)  # predicted_classes == labels.data is something like [1.jpg 0 1.jpg 1.jpg 1.jpg 0]
            running_corrects += num_correct_predictions
            running_loss += loss.item()

        epoch_loss = running_loss / len(training_loader.dataset)
        losses.append(epoch_loss)  # average loss of each epoch is added to the losses

        epoch_accuracy = running_corrects / len(training_loader.dataset)
        corrects.append(epoch_accuracy)

        finish_training_time = time.time()
        print('epoch:', e + 1, '... training loss:   {:.4f}'.format(epoch_loss), ', training accuracy:   {:.4f}'.format(epoch_accuracy)+ ' time passed: ' + str(timedelta(seconds=finish_training_time - start_training_time)))

        with torch.no_grad():
            for validation_images, validation_labels in validation_loader:
                validation_images = validation_images.to(device)
                validation_labels = validation_labels.to(device)

                validation_outputs = model.forward(validation_images)
                validation_loss = criterion(validation_outputs, validation_labels)

                _, validation_predicted_classes = torch.max(validation_outputs, 1)
                validation_num_correct_predictions = torch.sum(validation_predicted_classes == validation_labels.data)
                validation_running_corrects += validation_num_correct_predictions
                validation_running_loss += validation_loss.item()

            validation_epoch_loss = validation_running_loss / len(validation_loader.dataset)
            validation_losses.append(validation_epoch_loss)

            validation_epoch_accuracy = validation_running_corrects / len(validation_loader.dataset)
            validation_corrects.append(validation_epoch_accuracy)

            print('epoch:', e + 1, '... validation loss: {:.4f}'.format(validation_epoch_loss), ', validation accuracy: {:.4f}'.format(validation_epoch_accuracy),
                  '\n')

    if plot_loss_and_corrects:
        helper_functions.plot_loss_and_corrects_epoch(epochs, losses, corrects, validation_losses, validation_corrects)


# MAIN #################################################################################################################

def initialize_CNN():
    # helper functions class including some general purpose functions
    helper_functions = Helper_Functions()

    # Transforms for both training set and validation set
    # Data Augmentation (apply these transformations to training set only)
    transform_train = transforms.Compose([transforms.Resize((image_size, image_size)),  # resizes each image (pixels)
                                          transforms.RandomHorizontalFlip(),  # horizontal flip (lift to right)
                                          # random rotation hinders the performance
                                          transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Affine Type Transformations (stretch, scale)
                                          transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # changes color (this time, use 1.jpg)
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])

    # Transformations for Validation Set
    transform_validation = transforms.Compose([transforms.Resize((image_size, image_size)),
                                               transforms.ToTensor(),  # from (0, 255) intensity to (0, 1.jpg) probability
                                               transforms.Normalize((0.5,), (0.5,))])  # mean and center deviation to normalize (ranges from -1.jpg to 1.jpg)

    # Load dataset
    training_dataset = datasets.ImageFolder(dataset_path, transform=transform_train)
    validation_dataset = datasets.ImageFolder(dataset_path, transform=transform_validation)

    # obtain training and validation indices
    sample_num = len(training_dataset)
    sample_indices = list(range(sample_num))
    np.random.shuffle(sample_indices)
    split = int(np.floor((1 - train_validation_split_ratio) * sample_num))
    train_idx, valid_idx = sample_indices[split:], sample_indices[:split]

    # print info
    print('\n! Dataset from:', dataset_path, '\nTotal Image:', sample_num, '\nTrain-Validation Ratio:', train_validation_split_ratio, '\nTrain Image:', len(train_idx), '\nValidation Image:', len(valid_idx))
    print('! Classes:', training_dataset.classes, '\n')

    # define samplers for obtaining training and validation batches
    training_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    validation_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    # prepare data loaders
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, sampler=training_sampler)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, sampler=validation_sampler)

    # just to visualize dataset
    data_iter = iter(training_loader)
    pollen_images, labels = data_iter.next()

    pollen_images = pollen_images.to(device)
    labels = labels.to(device)

    if print_initial_dataset:
        helper_functions.show_images(pollen_images, labels, classes=training_dataset.classes)

    # get model
    model = CNN(training_dataset.classes, image_size)

    # create criterion and optimizer
    # nn.CrossEntropyLoss loss function is used for multiclass classification (requires raw output)
    # nn.CrossEntropyLoss is combination of log_softmax() and NLLLoss()
    criterion = nn.CrossEntropyLoss()

    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    train(model, training_loader, validation_loader, criterion, optimizer, helper_functions)
    torch.save(model, 'models/best_model.tf')
    print('! model saved.')


initialize_CNN()
