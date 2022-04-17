from .Pollen_Extraction import Pollen_Extraction
from .Helper_Functions import Helper_Functions

import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt


# ML Manager handles ML Subsystem
# Services:
# analyze_sample(self, sample_image, location=None, date=None, academic_name=None, erosion_dilation=10) -> source_img, analysis_text, pollens_dict
class ML_Manager:

    def __init__(self):

        # UPDATE for yourself
        self.model_path = '/Users/irem_/Documents/GitHub/CS491_Senior_Design_Project/code/api/ML_Subsystem/models/best_model.pth'
        #self.model_path = './models/best_model.pth'
        self.extractor = Pollen_Extraction()
        self.helper = Helper_Functions()

        # load model
        self.classes = ['acacia_dealbata', 'acer_negundo', 'ailanthus_altissima', 'alnus_glutinosa', 'ambrosia_artemisiifolia', 'betula_papyrifera', 'borago_officinalis',
                        'picea_abies']
        self.image_size = 300
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
        self.model.classifier[6] = nn.Linear(4096, len(self.classes))

        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        print('! model and state dict loaded.')

    def analyze_sample(self, sample_image, location=None, date=None, academic_name=None, erosion_dilation=10):
        # extract pollen images
        padding = 30
        square_threshold = 300
        square_dim_size = 300
        area_closing = 1000000
        plot_dilation = False
        plot_image = False
        plot_predicted = False

        pollen_images, box_coordinates = self.extractor.extract_PIL_Image(sample_image, padding, square_threshold, square_dim_size, n_dilation=erosion_dilation, area_closing=area_closing, plot_dilation=plot_dilation, plot_image=plot_image, plot_predicted=plot_predicted)

        # get predictions (forward to the model)
        pollens = []
        for img in pollen_images:
            # print(img)
            label = self.forward_image(img)
            # print('label:', label)
            pollens.append(label)

        # get pollen dict
        pollens_dict = dict()
        for i in pollens:
            pollens_dict[i] = pollens_dict.get(i, 0) + 1

        # get image with labels
        source_img = self.helper.label_sample_image(sample_image, box_coordinates, pollens=pollens, plot=True, no_grid=True)
        analysis_text = self.get_analysis_text(pollens_dict, location, date, academic_name)
        # source_img.save('pred.jpg')
        return source_img, analysis_text, pollens_dict

    def forward_image(self, img):

        transform_validation = transforms.Compose([transforms.Resize((self.image_size, self.image_size)),
                                                   transforms.ToTensor(),  # from (0, 255) intensity to (0, 1.jpg) probability
                                                   transforms.Normalize((0.5,), (0.5,))])  # mean and center deviation to normalize (ranges from -1.jpg to 1.jpg)
        img = transform_validation(img)
        img = img.unsqueeze(0)
        output = self.model.forward(img)
        _, predicted_classes = torch.max(output, 1)  # gets the maximum output value for each output
        return self.classes[predicted_classes.item()]

    def get_analysis_text(self, pollens_dict, location=None, date=None, academic_name=None):
        analysis_text = ''

        for pollen_name, count in pollens_dict.items():
            analysis_text += pollen_name + ' : ' + str(count) + '\n'

        return analysis_text

    # EXTRACTION
    # DO NOT TOUCH
    def extract_dataset_folder(self, source_directory, save_directory, error_directory, current_folder, dilation, area_closing, padding, square_threshold, square_dim_size, plot_threshold, plot_each, plot_final, plot_product, plot_dilation, save_each, reset_=False):
        self.extractor.extract_folder(source_directory, save_directory, error_directory, current_folder, dilation, area_closing, padding, square_threshold, square_dim_size, plot_threshold=plot_threshold, plot_each=plot_each, plot_final=plot_final, plot_product=plot_product, plot_dilation=plot_dilation, save_each=save_each, helper=self.helper, reset_=reset_)

    def dilation_erosion_test(self, source_directory, current_folder, dilation_range, area_closing, padding, square_threshold, square_dim_size, im_num=5, pass_num=0, plot=True, plot_dilation=False):
        self.extractor.dilation_erosion_test(source_directory, current_folder, dilation_range, area_closing, padding, square_threshold, square_dim_size, im_num, pass_num, plot, helper=self.helper, plot_dilation=plot_dilation)

    def send_SMS(self, text):
        self.helper.send_SMS(text)


# MAIN ############################################################################################################

#def main():
    #manager = ML_Manager()

    # # Analyze Sample
    #sample_image = Image.open("test_images/6.jpg")

    #source_img, analysis_text, pollens_dict = manager.analyze_sample(sample_image, erosion_dilation=10)

    #plt.imshow(source_img)
    #print('\n! Analysis text:\n', analysis_text)
    #print('! Pollens dictionary:\n', pollens_dict)


#if __name__ == "__main__":
#    main()
