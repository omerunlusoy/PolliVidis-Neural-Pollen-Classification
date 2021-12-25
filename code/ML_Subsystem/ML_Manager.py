import numpy as np
import matplotlib.pyplot as plt
import torch

from Pollen_Extraction import Pollen_Extraction
from PIL import Image, ImageDraw, ImageFont
from CNN import Pollen_Model


# ML Manager handles ML Subsystem
# analyze_sample(sample_image) -> PIL Image, analysis text
class ML_Manager:

    def __init__(self):
        self.extractor = Pollen_Extraction()
        self.model = torch.load('models/best_model')

    def analyze_sample(self, sample_image, dilation=10):

        pollen_image = Image.open("test_images/1.jpg")

        # extract pollen images
        pollen_images, box_coordinates = self.extractor.extract_PIL_Image(sample_image, dilation)

        # get predictions
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
        source_img = sample_image.convert("RGB")
        for i, coo in enumerate(box_coordinates):
            draw = ImageDraw.Draw(source_img)
            minr, minc, maxr, maxc = coo
            draw.rectangle([(minc, minr), (maxc, maxr)], outline='blue', width=12)
            # font = ImageFont.load_default()
            font = ImageFont.truetype("Other Implementations/Helvetica.ttc", 100)
            if minr - 100 > 0:
                draw.text((minc, minr - 100), pollens[i], font=font, fill='black')
            else:
                draw.text((minc, maxr + 30), pollens[i], font=font, fill='black')
        plt.imshow(source_img)
        plt.title('Labeled Image')
        # plt.savefig('prediction.jpg', dpi=500, bbox_inches='tight')
        plt.show()

        return pollen_image, self.get_analysis_text(pollens_dict), pollens_dict

    def get_analysis_text(self, pollens_dict):
        analysis_text = ''
        return analysis_text


##########################################################################################
sample_image = Image.open("test_images/pop.jpg")
manager = ML_Manager()
manager.analyze_sample(sample_image)
