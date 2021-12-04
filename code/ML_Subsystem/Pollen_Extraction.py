import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io, filters, morphology, color, segmentation
from skimage.color import rgb2gray, label2rgb
from skimage.measure import label, regionprops
from PIL import Image
import os


def get_image_and_threshold(file_name, plot=False):
    img = io.imread(file_name)  # Load Image
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


def binary_dilation(thresholded_img, n_dilation):
    image_dilated = thresholded_img  # OR binary_erosion
    for x in range(n_dilation):
        image_dilated = morphology.binary_dilation(image_dilated)
    return image_dilated


def label_image(dilated_img, gray_img, original_img, file_name, plot=False):
    # label image regions
    label_image = label(dilated_img)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_image, image=gray_img, bg_label=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    for i, region in enumerate(regionprops(label_image)):
        # take regions with large enough areas
        if region.area >= (300 * 300):
            # get segmented image
            get_segmented_image(region.coords, original_img, file_name, i, plot=plot)

            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    plt.tight_layout()
    plt.show()


def add_padding(ymax, xmax, ymin, xmin, yorg, xorg):

    padding = 10
    threshold = 200

    xlen = xmax - xmin
    ylen = ymax - ymin
    print(xmax, ymax, xmin, ymin, xlen, ylen, xorg, yorg)

    if abs(xlen - ylen) < threshold:

        if xlen < ylen:
            # increase x padding
            xpad = padding + (ylen - xlen)
            ypad = padding
        else:
            # increase y padding
            ypad = padding + (xlen - ylen)
            xpad = padding

        if int(xmin - xpad/2) < 0 or int(xmax + xpad/2) > xorg:
            # corner case
            if int(xmin - xpad/2) < 0:
                xpad -= xmin
                xmin = 0
                xmax += xpad
            else:
                print(xlen, ylen, xpad, ypad, xorg, yorg, xmax, xmin)
                xpad -= (yorg - xmax)
                xmax = yorg
                xmin -= xpad
        else:
            xmin -= math.ceil(xpad/2)
            xmax += math.floor(xpad/2)

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

        print(xmax, ymax, xmin, ymin, "a")
        return xmax, ymax, xmin, ymin, True
    return xmax, ymax, xmin, ymin, False


def get_segmented_image(coords, org_image, file_name, i, plot=False):
    arr = np.array(coords)
    xmax, ymax = arr.max(axis=0)
    xmin, ymin = arr.min(axis=0)
    arr2 = np.array(org_image)

    # add padding
    xmax, ymax, xmin, ymin, square = add_padding(xmax, ymax, xmin, ymin, len(arr2), len(arr2[0]))

    cropped = org_image[xmin:xmax, ymin:ymax]

    im = Image.fromarray(cropped)

    if square:
        im.save("cropped_images/" + str(file_name.title())[:-4] + "_" + str(i) + ".jpg")
    else:
        im.save("cropped_images/err/err_" + str(file_name.title())[:-4] + "_" + str(i) + ".jpg")
    if plot:
        # Show the results
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(org_image, 'gray')
        ax[1].imshow(cropped, 'gray')
        plt.show()


# MAIN ###################################################################################################

n_dilation = 16

directory = r'/Users/omerunlusoy/Desktop/CS 491/CS491_Senior_Design_Project/Ankara_Dataset/betula'
directory = r'/Users/omerunlusoy/Desktop/Coding/Python/Image_Extraction/cropped_images/err'

_, _, files = next(os.walk(directory))
file_count = len(files)


for i, filename in enumerate(os.listdir(directory)):

    if filename.endswith(".jpg"):

        file = os.path.join(directory, filename)

        original_img, gray_img, thresholded_img = get_image_and_threshold(file, plot=True)

        dilated_img = binary_dilation(thresholded_img, n_dilation)

        label_image(dilated_img, gray_img, original_img, filename, plot=True)

        print(str(i), "/", str(file_count), "completed.")

    else:
        continue
