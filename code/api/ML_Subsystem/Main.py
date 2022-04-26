from ML_Subsystem.ML_Manager import ML_Manager
from PIL import Image
import matplotlib.pyplot as plt

def main():

    manager = ML_Manager(load_model=True)

    # Test Image (PIL Image)
    test_image = Image.open("test_images/6.jpg")

    # Extraction Test (does not use model)
    morphology_sequence = '10'
    extraction_tested_image = manager.analyze_sample(test_image, location=None, date=None, academic_name=None, morphology_sequence=morphology_sequence, test_extraction=True)

    # Analyze Sample (all inputs except image is String)
    location = ''
    date = ''
    academic_name = ''
    morphology_sequence = '10'
    analyzed_img, analysis_text, pollens_dict = manager.analyze_sample(test_image, location=location, date=date, academic_name=academic_name, morphology_sequence=morphology_sequence, test_extraction=False)

    # Outputs
    plt.imshow(test_image)
    plt.axis('off')
    plt.show()
    plt.imshow(extraction_tested_image)
    plt.axis('off')
    plt.show()
    plt.imshow(analyzed_img)
    plt.axis('off')
    plt.show()
    print('\n! Analysis text:\n', analysis_text)
    print('! Pollens dictionary:\n', pollens_dict)


if __name__ == "__main__":
    main()
