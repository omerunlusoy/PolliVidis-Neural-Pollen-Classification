from skimage import filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2

# Setting plot size to 15, 15
plt.figure(figsize=(15, 15))

# Load Image
pollen_image = cv2.imread("4.jpg")

# Gray Scale
gray_image = rgb2gray(pollen_image)

# Computing Otsu's thresholding value
threshold = filters.threshold_otsu(gray_image)

# Computing binarized values using the obtained
# threshold
binarized_image = (gray_image > threshold) * 1

for i in range(len(binarized_image)):
    for ii in range(len(binarized_image[i])):
        if binarized_image[i][ii] == 0:
            pass

plt.subplot(2, 2, 1)
plt.title("Threshold: > " + str(threshold))

# Displaying the binarized image
plt.imshow(binarized_image, cmap="gray")

# Computing Ni black's local pixel
# threshold values for every pixel
threshold = filters.threshold_niblack(gray_image)

# Computing binarized values using the obtained
# threshold
binarized_image = (gray_image > threshold) * 1
plt.subplot(2, 2, 2)
plt.title("Niblack Thresholding")

# Displaying the binarized image
plt.imshow(binarized_image, cmap="gray")

# Computing Sauvola's local pixel threshold
# values for every pixel - Not Binarized
threshold = filters.threshold_sauvola(gray_image)
plt.subplot(2, 2, 3)
plt.title("Sauvola Thresholding")

# Displaying the local threshold values
plt.imshow(threshold, cmap="gray")

# Computing Sauvola's local pixel
# threshold values for every pixel - Binarized
binarized_image = (gray_image > threshold) * 1
plt.subplot(2, 2, 4)
plt.title("Sauvola Thresholding - Converting to 0's and 1.jpg's")

# Displaying the binarized image
plt.imshow(binarized_image, cmap="gray")

plt.show()

