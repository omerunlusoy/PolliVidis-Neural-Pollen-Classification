import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io, filters, morphology, color, segmentation
from skimage.color import rgb2gray, label2rgb
from skimage.measure import label, regionprops
from PIL import Image
import os
import time
from datetime import timedelta


# Pollen_Extraction class extract pollen images from sample images
# Services;
# extract_folder(source_directory, save_directory, current_folder, n_dilation)   # for dataset parsing
# extract_PIL_Image(PILImage, n_dilation=16)        # for single image, for user sample image

class Pollen_Extraction:

    def __init__(self):
        pass

    def get_image_and_threshold(self, file_name=None, PILImage=None, plot=False):
        if file_name is not None:
            img = io.imread(file_name)  # Load Image
        else:
            img = np.array(PILImage)

        image = rgb2gray(img)

        # threshold_otsu, threshold_niblack, threshold_sauvola
        threshold = filters.threshold_otsu(image)  # Calculate threshold
        image_thresholded = image < threshold  # Apply threshold

        if plot:
            # Show the results
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(img)
            ax[1].imshow(image, 'gray')
            ax[2].imshow(image_thresholded, 'gray')
            ax[0].set_title("Original")
            ax[1].set_title("Gray")
            ax[2].set_title("Thresholded")
            plt.show()
        return img, image, image_thresholded

    def binary_dilation(self, thresholded_img, n_dilation):
        image_dilated = thresholded_img  # OR binary_erosion
        for x in range(n_dilation):
            image_dilated = morphology.binary_dilation(image_dilated)
        return image_dilated

    def label_image(self, dilated_img, gray_img, original_img, file_name=None, save_folder=None, err_folder=None, plot=False):
        # label image regions
        label_image = label(dilated_img)
        # to make the background transparent, pass the value of `bg_label`,
        # and leave `bg_color` as `None` and `kind` as `overlay`
        image_label_overlay = label2rgb(label_image, image=gray_img, bg_label=0)
        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(image_label_overlay)

        ims = []
        for i, region in enumerate(regionprops(label_image)):
            # take regions with large enough areas
            if region.area >= (300 * 300):
                # get segmented image
                im = self.get_segmented_image(region.coords, original_img, file_name, i, save_folder, err_folder, plot=plot)
                if im is not None:
                    ims.append(im)
                if plot:
                    # draw rectangle around segmented coins
                    minr, minc, maxr, maxc = region.bbox
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
        if plot:
            plt.tight_layout()
            plt.show()
        return ims

    def get_segmented_image(self, coords, org_image, file_name, i, save_folder, err_folder, plot=False):
        arr = np.array(coords)
        ymax, xmax = arr.max(axis=0)
        ymin, xmin = arr.min(axis=0)
        arr2 = np.array(org_image)

        # add padding
        xmax, ymax, xmin, ymin, square = self.add_padding(xmax, ymax, xmin, ymin, len(arr2), len(arr2[0]))

        cropped = org_image[ymin:ymax, xmin:xmax]
        try:
            im = Image.fromarray(cropped)
        except:
            return None

        if square:
            if file_name is not None:
                file_name1 = str(file_name.title())[:-4] + "_" + str(i) + ".jpg"
                folder = os.path.join(save_folder, file_name1)
                im.save(folder)
            else:
                return im
        else:
            if file_name is not None:
                file_name1 = str(file_name.title())[:-4] + "_" + str(i) + ".jpg"
                folder = os.path.join(err_folder, file_name1)
                im.save(folder)
            else:
                return im
        if plot:
            # Show the results
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(org_image, 'gray')
            ax[1].imshow(cropped, 'gray')
            plt.show()

    def add_padding(self, xmax, ymax, xmin, ymin, yorg, xorg):
        padding = 10
        threshold = 200
        xlen = xmax - xmin
        ylen = ymax - ymin

        if abs(xlen - ylen) < threshold:
            if xlen < ylen:
                # increase x padding
                xpad = padding + (ylen - xlen)
                ypad = padding
            else:
                # increase y padding
                ypad = padding + (xlen - ylen)
                xpad = padding

            if int(xmin - xpad / 2) < 0 or int(xmax + xpad / 2) > xorg:
                # corner case
                if int(xmin - xpad / 2) < 0:
                    xpad -= xmin
                    xmin = 0
                    xmax += xpad
                else:
                    xpad -= (yorg - xmax)
                    xmax = yorg
                    xmin -= xpad
            else:
                xmin -= math.ceil(xpad / 2)
                xmax += math.floor(xpad / 2)

            if int(ymin - ypad / 2) < 0 or int(ymax + ypad / 2) > yorg:
                # corner case
                if int(ymin - ypad / 2) < 0:
                    ypad -= ymin
                    ymin = 0
                    ymax += ypad
                else:
                    ypad -= (xorg - ymax)
                    ymax = xorg
                    ymin -= ypad
            else:
                ymin -= math.ceil(ypad / 2)
                ymax += math.floor(ypad / 2)

            return xmax, ymax, xmin, ymin, True
        return xmax, ymax, xmin, ymin, False

    # for folder iteration
    def extract_image(self, file, filename, save_folder, err_folder, n_dilation, plot=False):
        original_img, gray_img, thresholded_img = self.get_image_and_threshold(file_name=file, plot=plot)
        dilated_img = self.binary_dilation(thresholded_img, n_dilation)
        self.label_image(dilated_img, gray_img, original_img, filename, save_folder, err_folder, plot=plot)

    # this function will be used to process images from users
    def extract_PIL_Image(self, PILImage, n_dilation=16):
        original_img, gray_img, thresholded_img = self.get_image_and_threshold(PILImage=PILImage, plot=True)
        dilated_img = self.binary_dilation(thresholded_img, n_dilation)
        return self.label_image(dilated_img, gray_img, original_img, plot=True)

    def extract_folder(self, source_directory, save_directory, current_folder, n_dilation, plot=False):
        # folders
        source_folder = os.path.join(source_directory, current_folder)
        save_folder = os.path.join(save_directory, current_folder)
        err_folder = os.path.join(save_folder, 'err')

        # create Ankara_Dataset_cropped/betula_cropped and Ankara_Dataset_cropped/betula_cropped/err
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
            if not os.path.exists(err_folder):
                os.mkdir(err_folder)

        # get file count
        _, _, files = next(os.walk(source_folder))
        file_count = len(files)

        start_training_time = time.time()
        for i, filename in enumerate(os.listdir(source_folder)):
            if filename.endswith(".jpg"):
                file = os.path.join(source_folder, filename)
                self.extract_image(file, filename, save_folder, err_folder, n_dilation, plot=plot)
                finish_training_time = time.time()
                print(str(i), "/", str(file_count), ':', filename, ', time passed: ' + str(timedelta(seconds=finish_training_time - start_training_time)))
            else:
                continue


# MAIN ###################################################################################################

def main():
    n_dilation = 1

    source_directory = r'/Users/omerunlusoy/Desktop/CS 491/CS491_Senior_Design_Project/Ankara_Dataset/'
    save_directory = r'/Users/omerunlusoy/Desktop/CS 491/CS491_Senior_Design_Project/Ankara_Dataset_cropped/'
    current_folder = r'populus'

    pollen_extraction = Pollen_Extraction()

    # extract folder for dataset
    pollen_extraction.extract_folder(source_directory, save_directory, current_folder, n_dilation, plot=True)

    # extract single image
    # pollen_image = Image.open("test_images/1.jpg")
    # img = pollen_extraction.extract_PIL_Image(pollen_image, 0)
    # for im in img:
    #     plt.imshow(im)
    #     plt.show()


if __name__ == "__main__":
    main()
