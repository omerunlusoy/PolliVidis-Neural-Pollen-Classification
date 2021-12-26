from .Pollen_Extraction import Pollen_Extraction
from .CNN import initialize_CNN, CNN
from .Helper_Functions import Helper_Functions

import torch


# ML Manager handles ML Subsystem
# analyze_sample(sample_image) -> PIL Image, analysis text
class ML_Manager:

    def __init__(self):
        self.extractor = Pollen_Extraction()
        self.helper = Helper_Functions()
        self.CNN = CNN()
        self.model = self.CNN.load_model()
        print('! model loaded.')

    def analyze_sample(self, sample_image, location, date, academic_name, db_manager, dilation=10):
        # extract pollen images
        pollen_images, box_coordinates = self.extractor.extract_PIL_Image(sample_image, dilation)

        # get predictions (forward to the CNN model)
        pollens = []
        for img in pollen_images:
            # print(img)
            label = self.model.forward_image(img)
            # print('label:', label)
            pollens.append(label)

        # get pollen dict
        pollens_dict = dict()
        for i in pollens:
            pollens_dict[i] = pollens_dict.get(i, 0) + 1

        # get image with labels
        source_img = self.helper.label_sample_image(sample_image, box_coordinates, pollens, plot=True)
        return source_img, self.get_analysis_text(pollens_dict, location, date, academic_name, db_manager), pollens_dict

    def extract_dataset_folder(self, source_directory, save_directory, current_folder, dilation):
        self.extractor.extract_folder(source_directory, save_directory, current_folder, dilation, plot=True)

    def get_analysis_text(self, pollens_dict, location, date, academic_name, db_manager=None):
        analysis_text = ''

        if db_manager is not None:
            for pollen_name, count in pollens_dict.items():
                text = db_manager.get_pollen_type(pollen_name).explanation_text
                analysis_text += pollen_name + ' : ' + count + '\n' + text + '\n'

        return analysis_text

    def train_model(self):
        initialize_CNN()


# MAIN ############################################################################################################

def main():

    pass
    # manager = ML_Manager()

    # # Analyze Sample
    # sample_image = Image.open("test_images/pop.jpg")
    # manager.analyze_sample(sample_image, -1, -1, -1, None)

    # extract dataset folder
    # source_directory = r'/Users/omerunlusoy/Desktop/CS 491/CS491_Senior_Design_Project/Ankara_Dataset/'
    # save_directory = r'/Users/omerunlusoy/Desktop/CS 491/CS491_Senior_Design_Project/Ankara_Dataset_cropped/'
    # current_folder = r'populus'
    # dilation = 1
    # manager.extract_dataset_folder(source_directory, save_directory, current_folder, dilation)

    # train model
    # manager.train_model()


if __name__ == "__main__":
    main()

