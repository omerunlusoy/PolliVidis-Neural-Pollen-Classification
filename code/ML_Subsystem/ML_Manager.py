import math
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import os
import time
from datetime import timedelta


# ML Manager handles ML Subsystem
# analyze_sample(sample_image) -> PIL Image, analysis text
class ML_Manager:

    def __init__(self):
        pass

    def analyze_sample(self, sample_image):
        analysis_text = ''
        pollen_image = Image.open("test_images/1.jpg")
        return pollen_image, analysis_text
