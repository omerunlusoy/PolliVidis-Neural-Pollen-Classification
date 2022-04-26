import torch
import torch.nn as nn
import torch.nn.functional as Functional
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

import requests                                                     # HTTP requests
import PIL.ImageOps
from PIL import Image                                               # Python Imaging Library

import os
import random
from datetime import datetime
from enum import Enum


# Enums ################################################################################################################

# Activation Enum
class Activation(Enum):
    Tanh = 1
    Relu = 2
    Leaky_Relu = 3
    Sigmoid = 4
    Soft_Plus = 5


# Initial_Weight Enum
class Initial_Weight(Enum):
    Xavier_Normal = 1
    Xavier_Uniform = 2
    Kaiming_Normal = 3
    Kaiming_Uniform = 4


class Sample_Method(Enum):
    Uniform = 1
    Lhs = 2
    Sobol = 3


# Variables ############################################################################################################

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # specifies run device for more optimum runtime

input_size = 2
output_size = 2
hidden_size = 25
residual_block_size = 6

lam = 1.0
epochs_Adam = 2000
epochs_LBFGS = 200
criterion = nn.MSELoss()

learning_rate = 1e-3
step_size = 200
gamma = 0.999

max_iter = 10
tolerance_grad = 1.e-12
tolerance_change = 1.e-15

seed = 1234

print_epoch_Adam_interval = 50
print_epoch_LBFGS_interval = 1

print_dataset = True
print_testset = True
plot_loss_and_corrects = True
train_anyway = True

save_path_name = "saved_models_Pollen"
save_model_name = "Pollen_model_"

log_filename = "log_Pollen.txt"


# Global Functions #####################################################################################################

def get_log_str(sample_method):
    log_str = str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "\ndevice:" + str(device) \
              + ", input_size:" + str(input_size) + ", output_size:" + str(output_size) + ", hidden_size:" + str(hidden_size) \
              + ", residual_block_size:" + str(residual_block_size) + ", sample_method:" + str(sample_method) \
              + ", lam:" + str(lam) + ", epochs_Adam:" + str(epochs_Adam) + ", epochs_LBFGS:" + str(epochs_LBFGS) \
              + ", learning_rate:" + str(learning_rate) + ", step_size:" + str(step_size) + ", gamma:" + str(gamma) + ", seed:" + str(seed)

    return log_str


def get_file_str(sample_method):
    filename_str = str(device) + str(input_size) + str(output_size) + str(hidden_size) + str(residual_block_size) + str(sample_method.value) \
                   + str(lam) + str(epochs_Adam) + str(epochs_LBFGS) + str(learning_rate) + str(step_size) + str(gamma)
    return filename_str


def activation(name):
    if name == Activation.Tanh:
        return nn.Tanh()
    elif name == Activation.Relu:
        return nn.ReLU(inplace=True)
    elif name == Activation.Leaky_Relu:
        return nn.LeakyReLU(inplace=True)
    elif name == Activation.Sigmoid:
        return nn.Sigmoid()
    elif name == Activation.Soft_Plus:
        return nn.Softplus()
    else:
        raise ValueError(f'unknown activation function: {name}')


def get_init_weight(name):
    if name == Initial_Weight.Xavier_Normal:
        return nn.init.xavier_normal_
    elif name == Initial_Weight.Xavier_Uniform:
        return nn.init.xavier_uniform_
    elif name == Initial_Weight.Kaiming_Normal:
        return nn.init.kaiming_normal_
    elif name == Initial_Weight.Kaiming_Uniform:
        return nn.init.kaiming_uniform_
    else:
        raise ValueError(f'unknown initialization weight: {name}')


def enter_log(text, log_str, filename_str, header=False):
    file = open(log_filename, "a")
    if header:
        file.write("\n" + log_str + "\n" + "Filename: " + filename_str + "\n")
    file.write(text + "\n")
    file.close()


def save_model(model, filename_str):
    # Path to save the model
    if not os.path.exists(save_path_name):
        os.mkdir(save_path_name)

    path = os.path.join(save_path_name, filename_str)
    if not os.path.isfile(path):
        torch.save(model, path)


def get_model(filename_str):
    if not os.path.exists(save_path_name):
        return None

    path = os.path.join(save_path_name, filename_str)
    if not os.path.isfile(path):
        return None

    model = torch.load(path)
    return model


# derivative of y respect to x
def grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)


def numpy_to_tensor(x):
    lst = []
    for item in x:
        lst.append(torch.from_numpy(np.asarray(item)).float())
    return lst


def assign_to_device(x):
    lst = []
    for item in x:
        if not isinstance(item, list):
            lst.append(item.to(device))
        else:
            lst.append(assign_to_device(item))
    return tuple(lst)


def plot_loss_epoch(epochs, losses, validation_losses, method):

    plt.title(f'Epoch vs Loss ({method})')
    plt.plot(range(epochs), losses, label="training loss of " + str(method))
    plt.plot(np.array(validation_losses)[:, 0], np.array(validation_losses)[:, 1], label="validation loss of " + str(method))
    # plt.xlabel('epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(color='gray', linestyle='-', linewidth=0.2)
    # plt.show()

# Residual Block Class #################################################################################################

class Residual_Block(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, activation_name=Activation.Tanh):
        super().__init__()
        self.activation1 = activation(activation_name)
        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.activation2 = activation(activation_name)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)

    def forward(self, x):
        i = x
        x = self.linear1(self.activation1(x))
        x = self.linear2(self.activation2(x))
        return i + x

# Residual DNN Class ###################################################################################################

class Residual_DNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, residual_blocks, activation_name=Activation.Tanh, init_name=Initial_Weight.Kaiming_Normal):

        super().__init__()

        model = nn.Sequential()
        model.add_module('fc_first', nn.Linear(input_size, hidden_size, bias=True))

        for i in range(residual_blocks):
            residual_block = Residual_Block(hidden_size, hidden_size, hidden_size, activation_name=activation_name)
            model.add_module(f'res_block{i + 1}', residual_block)

        model.add_module('act_last', activation(activation_name))
        model.add_module('fc_last', nn.Linear(hidden_size, output_size, bias=True))

        self.model = model
        self.init_weight(init_name)

    def forward(self, x):
        return self.model(x)

    def init_weight(self, name):
        nn_init = get_init_weight(name)

        for param in self.parameters():
            if len(param.shape) > 1:
                nn_init(param)

    def model_size(self):
        n_params = 0
        for param in self.parameters():
            n_params += param.numel()
        return n_params

# MAIN #################################################################################################################

def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # transform image
    transform = transforms.Compose([transforms.Resize((28, 28)),
                                    transforms.ToTensor(),  # from (0, 255) intensity to (0, 1.jpg) probability
                                    transforms.Normalize((0.5,), (0.5,))])  # mean and center deviation to normalize (ranges from -1.jpg to 1.jpg)

    # training dataset
    training_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)  # shuffle not to stuck in a local minimum

    data_iter = iter(training_loader)
    images, labels = data_iter.next()

    images = images.to(device)
    labels = labels.to(device)

    if print_initial_dataset:
        show_images(images, labels)

    # validation dataset (Test Model)
    validation_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)  # no need to shuffle

    model_conv = get_model()
    if model_conv is None or train_anyway:
        model_conv = LeNet().to(device=device)

        # nn.CrossEntropyLoss loss function is used for multiclass classification (requires raw output)
        # nn.CrossEntropyLoss is combination of log_softmax() and NLLLoss()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model_conv.parameters(), lr=learning_rate)

        train_network(model_conv, training_loader, criterion, optimizer)

        # save model
        save_model(model_conv)

    # plot predictions
    data_iter = iter(validation_loader)
    images, labels = data_iter.next()

    images = images.to(device)
    labels = labels.to(device)

    outputs = model_conv.forward(images)
    _, predicted_classes = torch.max(outputs, 1)

    if print_testset:
        show_images(images, labels, predicted_classes)










    # holds losses for each method
    losses_method = {}

    # determine sample method
    for sample_method in Sample_Method:
        print('\nTraining with sample method:', sample_method)
        enter_log("\ntraining with sample method:" + str(sample_method), get_log_str(sample_method), get_file_str(sample_method), True)

        model = get_model(get_file_str(sample_method))
        if model is None or train_anyway:
            # create model
            model = Residual_PINN_Schrodinger(input_size, hidden_size, output_size, residual_block_size, activation_name=Activation.Tanh, init_name=Initial_Weight.Xavier_Normal)
            # print(model)
            enter_log("Model initialized.", get_log_str(sample_method), get_file_str(sample_method))

            # trainset
            trainset = Trainset_Schrodinger(problem, method=sample_method, N_x=n_x, N_y=n_y, N=n, N_boundary=n_bc, purpose='Training Set')
            # print(trainset.get_trainset())
            enter_log("Trainset is created.", get_log_str(sample_method), get_file_str(sample_method))

            # has to be Lhs
            validationset = Trainset_Schrodinger(problem, method=Sample_Method.Lhs, N=100, N_boundary=100, purpose='Validation Set')
            # print("a:", validationset.get_trainset())
            enter_log("Validationset is created.", get_log_str(sample_method), get_file_str(sample_method))

            # trainer
            trainer = Trainer_Schrodinger(problem, model, trainset, validationset, sample_method)
            enter_log("Training begins...", get_log_str(sample_method), get_file_str(sample_method))

            # save losses
            losses, validation_losses = trainer.train()
            losses_method[sample_method] = (losses, validation_losses)

            # save model
            save_model(model, get_file_str(sample_method))
            enter_log("Model saved.", get_log_str(sample_method), get_file_str(sample_method))
        else:
            print("model is loaded from a saved file.")


        # testset
        testset = Testset_Schrodinger(problem, method=sample_method, N_x=n_x, N_y=n_y, N=n, N_boundary=n_bc)
        # X, t, x, Exact_h = testset.get_testset()
        # print('X:', X, '\nt:', t, '\nx:', x, '\nExact_h', Exact_h)

        # tester
        tester = Tester_Schrodinger(problem, model, testset)

        # plot tests
        tester.plot(sample_method)

        # test model
        tester.test(sample_method)

    # plot losses
    if len(losses_method) is not 0:
        for key, value in losses_method.items():
            (losses, validation_losses) = value
            plot_loss_epoch(epochs_Adam+epochs_LBFGS, losses, validation_losses, key)

        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(color='gray', linestyle='-', linewidth=0.2)
        plt.xlim(0, 150)
        plt.ylim(0, 10)
        # plt.savefig('loss_comparison.jpg', dpi=500, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    main()

