import math
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io, filters, morphology, color, segmentation
from skimage.color import rgb2gray, label2rgb
from skimage.measure import label, regionprops
from PIL import Image, ImageDraw, ImageFont
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

    # this function will be used to process images from users
    def extract_PIL_Image(self, PILImage, padding, square_threshold, square_dim_size, n_dilation=16, area_closing=1000000, plot_dilation=False, plot_image=True, plot_predicted=True):
        original_img, gray_img, thresholded_img = self.get_image_and_threshold(PILImage=PILImage, plot=plot_image)
        dilated_img = self.binary_dilation_or_erosion(thresholded_img, n_dilation, plot_dilation=plot_dilation, area_closing=area_closing)
        labeled_images, box_coordinates = self.label_image(dilated_img, gray_img, original_img, padding, square_threshold, square_dim_size, plot=plot_predicted, save_each=True)
        return labeled_images, box_coordinates

    def extract_folder(self, source_directory, save_directory, error_directory, current_folder, n_dilation, area_closing, padding, square_threshold, square_dim_size, plot_threshold, plot_each, plot_final, plot_product, plot_dilation, save_each, helper, reset_=False):
        # folders
        source_folder = os.path.join(source_directory, current_folder)
        save_folder = os.path.join(save_directory, current_folder)
        err_folder = os.path.join(error_directory, current_folder)

        # create Ankara_Dataset_cropped/betula_cropped and Ankara_Dataset_cropped/betula_cropped/err
        if reset_:
            shutil.rmtree(save_folder)
            shutil.rmtree(err_folder)
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
                self.extract_image(file, filename, save_folder, err_folder, n_dilation, area_closing, padding, square_threshold, square_dim_size, plot_threshold=plot_threshold, plot_each=plot_each, plot_final=plot_final, plot_product=plot_product, plot_dilation=plot_dilation, fig_title='', save_each=save_each, helper=helper)
                finish_training_time = time.time()
                if i % 10 == 0:
                    print(str(i), "/", str(file_count), ':', filename, ', time passed: ' + str(timedelta(seconds=finish_training_time - start_training_time)))
            else:
                continue

    def dilation_erosion_test(self, source_directory, current_folder, dilation_range, area_closing, padding, square_threshold, square_dim_size, im_num=5, pass_num=0, plot=True, save_each=False, plot_dilation=False, helper=None):
        # folders
        source_folder = os.path.join(source_directory, current_folder)

        for i, filename in enumerate(os.listdir(source_folder)):
            if i < pass_num:
                continue
            for n_dilation in range(dilation_range[0], dilation_range[1], dilation_range[2]):
                if i >= im_num + pass_num:
                    break
                elif filename.endswith(".jpg"):
                    file = os.path.join(source_folder, filename)
                    fig_title = str(current_folder) + ', Dilation: ' + str(n_dilation)
                    if n_dilation == dilation_range[0]:
                        plot_ = plot
                    else:
                        plot_ = False
                    self.extract_image(file, '', '', '', n_dilation, area_closing, padding, square_threshold, square_dim_size, plot_threshold=plot_, plot_each=False, plot_final=False, plot_product=True, plot_dilation=plot_dilation, fig_title=fig_title, save_each=save_each, helper=helper)
                else:
                    continue

    # for folder iteration
    def extract_image(self, file, filename, save_folder, err_folder, n_dilation, area_closing, padding, square_threshold, square_dim_size, plot_threshold=False, plot_each=False, plot_final=False, plot_product=False, plot_dilation=False, fig_title='', save_each=True, helper=None):
        original_img, gray_img, thresholded_img = self.get_image_and_threshold(file_name=file, plot=plot_threshold, fig_title=fig_title)
        dilated_img = self.binary_dilation_or_erosion(thresholded_img, n_dilation, filename=file, plot_dilation=plot_dilation, area_closing=area_closing)
        ims, box_coordinates = self.label_image(dilated_img, gray_img, original_img, padding, square_threshold, square_dim_size, filename, save_folder, err_folder, plot=plot_final, plot_each=plot_each, fig_title=fig_title, save_each=save_each)
        if helper:
            plt.show()
            sample_image = Image.open(file)
            helper.label_sample_image(sample_image, box_coordinates, pollens=None, plot=plot_product, title=filename.rpartition('/')[-1] + ',   Erosion/Dilation: ' + str(n_dilation))

    def binary_dilation_or_erosion(self, thresholded_img, n_dilation, filename='', plot_dilation=False, area_closing=1000000):
        image_dilated = thresholded_img  # OR binary_erosion
        n_dilation = np.abs(n_dilation)

        # n erosion
        for _ in range(n_dilation):
            image_dilated = morphology.binary_erosion(image_dilated)

        # n dilation
        for _ in range(n_dilation):
            image_dilated = morphology.binary_dilation(image_dilated)

        # area closing
        image_dilated = morphology.area_closing(image_dilated, area_closing)

        if plot_dilation:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(thresholded_img)
            ax.set_title('Original')
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(image_dilated)
            ax.set_title('Erosion+Dilation+Area Closing with ' + str(str(n_dilation)))
            fig.suptitle(filename.rpartition('/')[-1] + ',   Erosion/Dilation: ' + str(n_dilation))

        return image_dilated

    def get_image_and_threshold(self, file_name=None, PILImage=None, plot=False, fig_title=''):
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
            fig.suptitle(fig_title)
            ax[0].imshow(img)
            ax[1].imshow(image, 'gray')
            ax[2].imshow(image_thresholded, 'gray')
            ax[0].set_title("Original")
            ax[1].set_title("Gray")
            ax[2].set_title("Thresholded")
            # plt.savefig('threshold.jpg', dpi=500, bbox_inches='tight')
            plt.axis('off')
            plt.show()
        return img, image, image_thresholded

    def label_image(self, dilated_img, gray_img, original_img, padding, square_threshold, square_dim_size, file_name=None, save_folder=None, err_folder=None, plot=False, plot_each=False, fig_title='', save_each=True):
        # label image regions
        label_image = label(dilated_img)
        # to make the background transparent, pass the value of `bg_label`,
        # and leave `bg_color` as `None` and `kind` as `overlay`
        image_label_overlay = label2rgb(label_image, image=gray_img, bg_label=0)

        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(fig_title)
            ax.imshow(image_label_overlay)

        ims = []
        box_coordinates = []
        for i, region in enumerate(regionprops(label_image)):
            # take regions with large enough areas
            if region.area >= (square_dim_size * square_dim_size):
                # get segmented image
                im = self.get_segmented_image(region.coords, original_img, file_name, i, save_folder, err_folder, padding, square_threshold, plot=plot_each, fig_title=fig_title, save_each=save_each)
                if im is not None:
                    ims.append(im)

                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                box_coordinates.append((minr, minc, maxr, maxc))
                if plot_each:
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
        if plot_each:
            plt.tight_layout()
            plt.savefig('label.jpg', dpi=500, bbox_inches='tight')
            plt.show()
        return ims, box_coordinates

    def get_segmented_image(self, coords, org_image, file_name, i, save_folder, err_folder, padding, square_threshold, plot=False, fig_title='', save_each=True):
        arr = np.array(coords)
        ymax, xmax = arr.max(axis=0)
        ymin, xmin = arr.min(axis=0)
        arr2 = np.array(org_image)

        # add padding
        xmax, ymax, xmin, ymin, square = self.add_padding(xmax, ymax, xmin, ymin, len(arr2), len(arr2[0]), padding=padding, square_threshold=square_threshold)

        cropped = org_image[ymin:ymax, xmin:xmax]
        try:
            im = Image.fromarray(cropped)
        except:
            return None

        if save_each:
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
            fig.suptitle(fig_title)
            plt.show()

    def add_padding(self, xmax, ymax, xmin, ymin, yorg, xorg, padding=10, square_threshold=200):

        xlen = xmax - xmin
        ylen = ymax - ymin

        if abs(xlen - ylen) < square_threshold:
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

    def binary_dilation_or_erosion_old(self, thresholded_img, n_dilation):
        image_dilated = thresholded_img  # OR binary_erosion
        if n_dilation >= 0:
            for _ in range(n_dilation):
                image_dilated = morphology.binary_dilation(image_dilated)
        else:
            n_dilation = -n_dilation
            for _ in range(n_dilation):
                image_dilated = morphology.binary_erosion(image_dilated)

        # image_dilated = morphology.closing(image_dilated)

        return image_dilated