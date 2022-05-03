import matplotlib.pyplot as plt
from .Pollen_Extraction import Pollen_Extraction
from .Helper_Functions import Helper_Functions
from .Paths_and_Keys import Paths_and_Keys
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms


# ML Manager handles ML Subsystem
# analyze_sample(sample_image) -> PIL Image, analysis text
class ML_Manager:

    def __init__(self, load_model=True):
        self.extractor = Pollen_Extraction()
        self.helper = Helper_Functions()
        self.paths_and_keys = Paths_and_Keys()

        if load_model:
            # load model
            self.classes = ['acacia_dealbata', 'acer_negundo', 'ailanthus_altissima', 'alnus_glutinosa', 'ambrosia_artemisiifolia', 'artemisia_vulgaris', 'betula_papyrifera', 'borago_officinalis', 'carpinus_betulus', 'chenopodium_album', 'cichorium_intybus', 'juglans_regia', 'juniperus_communis', 'ligustrum_robustrum', 'olea_europaea', 'phleum_phleoides', 'picea_abies', 'populus_nigra', 'quercus_robur', 'rubia_peregrina', 'rumex_stenophyllus', 'thymbra_spicata', 'ulmus_minor']

            self.image_size = 200
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
            self.model.classifier[6] = nn.Linear(4096, len(self.classes))


            self.model.load_state_dict(torch.load(self.paths_and_keys.model_path, map_location=torch.device('cpu')))
            self.model.eval()
            print('! model and state dict loaded.')

    def forward_image(self, img):

        transform_validation = transforms.Compose([transforms.Resize((self.image_size, self.image_size)),
                                                   transforms.ToTensor(),  # from (0, 255) intensity to (0, 1.jpg) probability
                                                   transforms.Normalize((0.5,), (0.5,))])  # mean and center deviation to normalize (ranges from -1.jpg to 1.jpg)
        img = transform_validation(img)
        img = img.unsqueeze(0)
        output = self.model.forward(img)
        _, predicted_classes = torch.max(output, 1)  # gets the maximum output value for each output
        return self.classes[predicted_classes.item()]

    def analyze_sample(self, sample_image, location=None, date=None, academic_name=None, morphology_sequence=None, test_extraction=False):
        # extract pollen images
        padding = 50
        square_threshold = 400
        square_dim_size = 100
        area_closing = 1000000
        plot_dilation = False
        plot_image = False
        plot_predicted = False

        erosion_dilation_ = None
        morphology_sequence_ = None

        if morphology_sequence == '' or morphology_sequence is None:
            erosion_dilation_ = 10
        elif morphology_sequence.isdigit():
            erosion_dilation_ = int(morphology_sequence)
        else:
            morphology_sequence_ = morphology_sequence

        if test_extraction:
            pollen_images, box_coordinates = self.extractor.extract_PIL_Image(sample_image, padding, square_threshold, square_dim_size, n_dilation=erosion_dilation_,
                                                                              area_closing=area_closing, plot_dilation=plot_dilation, plot_image=plot_image,
                                                                              plot_predicted=plot_predicted, morphology_sequence=morphology_sequence_)
            final_img = self.helper.label_sample_image(sample_image, box_coordinates, pollens=None, plot=False, no_grid=True, Helvetica_path_=self.paths_and_keys.Helvetica_path)
            return final_img

        pollen_images, box_coordinates = self.extractor.extract_PIL_Image(sample_image, padding, square_threshold, square_dim_size, n_dilation=erosion_dilation_, area_closing=area_closing, plot_dilation=plot_dilation, plot_image=plot_image, plot_predicted=plot_predicted, morphology_sequence=morphology_sequence_)

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
        final_img = self.helper.label_sample_image(sample_image, box_coordinates, pollens=pollens, plot=False, no_grid=True, Helvetica_path_=self.paths_and_keys.Helvetica_path)
        analysis_text = self.get_analysis_text(pollens_dict, location, date, academic_name)
        return final_img, analysis_text, pollens_dict

    def get_analysis_text(self, pollens_dict, location=None, date=None, academic_name=None):
        analysis_text = ''

        for pollen_name, count in pollens_dict.items():
            analysis_text += pollen_name + ' : ' + str(count) + '\n'

        return analysis_text

    # EXTRACTION
    def extract_dataset_folder(self, source_directory, save_directory, error_directory, current_folder, dilation, area_closing, padding, square_threshold, square_dim_size, plot_threshold, plot_each, plot_final, plot_product, plot_dilation, save_each, reset_=False, error_correction=False, error_padding=200, morphology_sequence=None):
        self.extractor.extract_folder(source_directory, save_directory, error_directory, current_folder, dilation, area_closing, padding, square_threshold, square_dim_size, plot_threshold=plot_threshold, plot_each=plot_each, plot_final=plot_final, plot_product=plot_product, plot_dilation=plot_dilation, save_each=save_each, helper=self.helper, reset_=reset_, error_correction=error_correction, error_padding=error_padding, morphology_sequence=morphology_sequence, Helvetica_path_=self.paths_and_keys.Helvetica_path)

    def dilation_erosion_test(self, source_directory, current_folder, dilation_range, area_closing, padding, square_threshold, square_dim_size, im_num=5, pass_num=0, plot=True, plot_dilation=False, morphology_sequence=None):
        self.extractor.dilation_erosion_test(source_directory, current_folder, dilation_range, area_closing, padding, square_threshold, square_dim_size, im_num, pass_num, plot, helper=self.helper, plot_dilation=plot_dilation, morphology_sequence=morphology_sequence, Helvetica_path_=self.paths_and_keys.Helvetica_path)

    def send_SMS(self, text):
        self.helper.send_SMS(text)

    def dataset_extraction_procedure(self):
        # extract dataset folder
        source_directory = self.paths_and_keys.dataset_source_path
        save_directory = self.paths_and_keys.dataset_save_path
        error_directory = self.paths_and_keys.dataset_error_path

        # for folder extraction
        folder_dict = {

        }

        done_folder_dict = {
            r'ambrosia_artemisiifolia': [20, 30, 200, 300, 1000000, 'E20-D20-AC1000000'],
            r'alnus_glutinosa': [40, 70, 200, 400, 1000000, 'E40-D40-AC1000000'],
            r'betula_papyrifera': [40, 50, 200, 400, 1000000, 'E40-D40-AC1000000'],
            r'acer_negundo': [40, 50, 200, 300, 1000000, 'E40-D40-AC1000000'],
            r'picea_abies': [5, 70, 400, 400, 1000000, 'E5-D5-AC1000000'],
            r'acacia_dealbata': [5, 70, 400, 400, 1000000, 'E5-D5-AC1000000'],
            r'ailanthus_altissima': [5, 50, 200, 200, 1000000, 'E5-D5-AC1000000'],
            r'borago_officinalis': [10, 50, 300, 200, 1000000, 'E10-D10-AC1000000'],
            r'carpinus_betulus': [10, 70, 300, 300, 100000, 'E10-D10-AC100000'],
            r'cichorium_intybus': [30, 80, 300, 300, 1000000, 'E30-D30-AC1000000'],
            r'ligustrum_robustrum': [10, 60, 300, 300, 1000000, 'E10-D10-AC1000000'],
            r'olea_europaea': [10, 40, 200, 200, 100000, 'E10-D10-AC100000'],
            r'thymbra_spicata': [15, 80, 300, 300, 500000, 'E15-D15-AC500000'],
            r'ulmus_minor': [10, 80, 300, 300, 500000, 'E10-D10-AC500000'],
            r'populus_nigra': [5, 60, 300, 300, 2000000, 'E5-D5-AC2000000'],
            r'phleum_phleoides': [5, 80, 400, 300, 500000, 'E5-D5-AC500000'],
            r'juniperus_communis': [5, 60, 200, 200, 500000, 'E5-D5-AC500000'],
            r'chenopodium_album': [20, 30, 200, 200, 500000, 'E20-D20-AC500000'],
            r'artemisia_vulgaris': [12, 80, 200, 200, 1000000, 'E12-D12-AC1000000'],
            r'juglans_regia': [-1, 100, 300, 300, 1000000, 'AC1000000-E15-D15'],
            r'quercus_robur': [-1, 100, 300, 300, 1000000, 'D7-E7-AC10000'],
            r'rubia_peregrina': [-1, 60, 200, 200, 1000000, 'AC10000-E10-D10'],
            r'rumex_stenophyllus': [-1, 100, 300, 300, 1000000, 'D10-E10-AC1000000'],
        }

        # Dilation Testing
        dilation_testing = False
        error_correction_ = True

        # Image Parameters
        # erosion_dilation = 10
        padding = 80  # default 20
        square_threshold = 200  # default 200
        square_dim_size = 200  # default 300
        area_closing = 1000000  # default 1000000

        dilation_range = [2, 6, 5]

        current_folders = [r'alnus_glutinosa', r'betula_papyrifera', r'acer_negundo', r'borago_officinalis', r'picea_abies', r'cichorium_intybus', r'ligustrum_robustrum',
                           r'ulmus_minor', r'phleum_phleoides', r'populus_nigra', r'artemisia_vulgaris']

        morphology_sequence1 = ['E5-D5-AC1000000', 'E10-D10-AC1000000', 'E15-D15-AC1000000', 'E20-D20-AC1000000',
                               'AC1000000-E5-D5', 'AC1000000-E10-D10', 'AC1000000-E15-D15', 'AC1000000-E20-D20',
                               'D5-E5-AC1000000', 'D10-E10-AC1000000', 'D15-E15-AC1000000', 'D20-E20-AC1000000',
                               'AC1000000-D5-E5', 'AC1000000-D10-E10', 'AC1000000-D15-E15', 'AC1000000-D20-E20']

        morphology_sequence2 = ['E20-D20-AC1000000', 'E30-D30-AC1000000']
        morphology_sequence3 = ['C1']

        morphology_sequence = morphology_sequence1

        if dilation_testing:
            for current_folder in current_folders:
                plot_dilation = True
                im_num = 8
                pass_num = 2
                self.dilation_erosion_test(source_directory, current_folder, dilation_range, area_closing, padding, square_threshold, square_dim_size, im_num=im_num,
                                              pass_num=pass_num, plot=False, plot_dilation=plot_dilation, morphology_sequence=morphology_sequence)

        elif error_correction_:
            for current_folder in current_folders:
                erosion_dilation = -1
                padding = done_folder_dict[current_folder][1]
                square_threshold = done_folder_dict[current_folder][2]
                square_dim_size = done_folder_dict[current_folder][3]
                area_closing = done_folder_dict[current_folder][4]
                morphology_sequence = 'D1-E1-AC10'

                plot_threshold = True
                plot_each = False
                plot_final = True
                plot_dilation = True
                plot_product = True
                save_each = True
                reset_ = False

                error_correction = True
                error_padding = 200

                save_directory = error_directory + current_folder + '/'
                self.extract_dataset_folder(error_directory, save_directory, save_directory, current_folder, erosion_dilation, area_closing, padding, square_threshold,
                                               square_dim_size, plot_threshold, plot_each, plot_final, plot_product, plot_dilation, save_each, reset_=reset_,
                                               error_correction=error_correction, error_padding=error_padding, morphology_sequence=morphology_sequence)
                self.extractor.rename(folder_name=current_folder, directory=error_directory)

        else:
            for current_folder in folder_dict:

                erosion_dilation = folder_dict[current_folder][0]
                padding = folder_dict[current_folder][1]
                square_threshold = folder_dict[current_folder][2]
                square_dim_size = folder_dict[current_folder][3]
                area_closing = folder_dict[current_folder][4]

                morphology_sequence = None
                if len(folder_dict[current_folder]) > 5:
                    morphology_sequence = folder_dict[current_folder][5]

                plot_threshold = False
                plot_each = False
                plot_final = False
                plot_dilation = False
                plot_product = False
                save_each = True
                reset_ = False

                try:
                    print('\n', current_folder, 'extraction started...')
                    self.extract_dataset_folder(source_directory, save_directory, error_directory, current_folder, erosion_dilation, area_closing, padding, square_threshold,
                                                   square_dim_size, plot_threshold, plot_each, plot_final, plot_product, plot_dilation, save_each, reset_=reset_,
                                                   morphology_sequence=morphology_sequence)
                    self.send_SMS(current_folder + ' extraction finished.')
                except:
                    self.send_SMS(current_folder + ' extraction raised error!!!.')
                    continue

            if not dilation_testing and not error_correction_:
                try:
                    self.send_SMS('execution finished.')
                except:
                    print('unable to send SMS!!!')
