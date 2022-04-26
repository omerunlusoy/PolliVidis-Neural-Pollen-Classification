import os
from enum import Enum

class User(Enum):
    Omer = 1
    Irem = 2
    Umut_Ada = 3
    Gamze = 4
    Ece = 5
    Other = 6


# current user and model
current_user = User.Omer
current_model_name = 'AlexNet_96.69_model.pth'


class Paths_and_Keys:

    # ML_Subsystem_path is mandatory, datasets_path is for Omer
    def __init__(self):
        if current_user == User.Omer:
            self.ML_Subsystem_path = r'/Users/omerunlusoy/Desktop/CS 492/copy_project/ML_Subsystem/'
            self.datasets_path = r'/Users/omerunlusoy/Desktop/CS 492/PolliVidis-Neural-Pollen-Classification/datasets/'
            self.example_image_for_GUI_path = r'/Users/omerunlusoy/Desktop/CS 492/PolliVidis-Neural-Pollen-Classification/datasets/Ankara_Dataset/populus_nigra/1.jpg'

        elif current_user == User.Irem:
            self.ML_Subsystem_path = r''
            self.datasets_path = r''
            self.example_image_for_GUI_path = r''

        elif current_user == User.Umut_Ada:
            self.ML_Subsystem_path = r''
            self.datasets_path = r''
            self.example_image_for_GUI_path = r''

        elif current_user == User.Gamze:
            self.ML_Subsystem_path = r''
            self.datasets_path = r''
            self.example_image_for_GUI_path = r''

        elif current_user == User.Ece:
            self.ML_Subsystem_path = r''
            self.datasets_path = r''
            self.example_image_for_GUI_path = r''

        elif current_user == User.Other:
            self.ML_Subsystem_path = r''
            self.datasets_path = r''
            self.example_image_for_GUI_path = r''

        # DO NO TOUCH
        self.model_path = os.path.join(self.ML_Subsystem_path, 'models/' + current_model_name)
        self.Helvetica_path = os.path.join(self.ML_Subsystem_path, 'Other_Implementations/Helvetica.ttc')
        self.dataset_source_path = os.path.join(self.datasets_path, 'Ankara_Dataset/')
        self.dataset_save_path = os.path.join(self.datasets_path, 'Ankara_Dataset_cropped/')
        self.dataset_error_path = os.path.join(self.datasets_path, 'Ankara_Dataset_cropped/error/')
