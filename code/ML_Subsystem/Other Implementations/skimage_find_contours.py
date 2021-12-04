import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, measure, morphology, color
from skimage.color import rgb2gray


test_sample = io.imread("2.jpg")
test_sample = rgb2gray(test_sample)

# Find contours at a constant value of 10
contours = measure.find_contours(image=test_sample, fully_connected='low')

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(test_sample, cmap=plt.cm.gray)

for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

fig, ax = plt.subplots()

threshold = 100

for contour in contours:
    con = contour.astype(int)
    start_row = min(con[:, 0])
    end_row = max(con[:, 0])
    start_col = min(con[:, 1])
    end_col = max(con[:, 1])

    size = (end_col - start_col) * (end_row - start_row)
    # print((end_col - start_col), (end_row - start_row), size)

    if (end_col - start_col) > threshold and (end_row - start_row) > threshold:
        ax.imshow(test_sample[start_row:end_row, start_col:end_col], cmap=plt.cm.gray)
        ax.axis('image')
        ax.set_title(str((end_col - start_col)) + " " + str((end_row - start_row)))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()


test_sample_smoothed = filters.median(test_sample, selem=np.ones((5, 5)))
