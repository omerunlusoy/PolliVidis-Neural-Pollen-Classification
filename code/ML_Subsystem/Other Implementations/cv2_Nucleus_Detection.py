import cv2
import numpy as np
from skimage import io, filters, morphology, color


class SelectNucleus:
    def __init__(self):
        """Load image etc."""
        self.img = io.imread("acer.jpg")  # Load Image
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        self.img = np.array(self.img, dtype=np.uint8)

    def split_nucleus(self):
        """split image and into few parts, one neclues in each"""
        _, thresh = cv2.threshold(self.img, 100, 255, 0)
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        kernel = np.ones((10, 10), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        cv2.imshow('a', self.img)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        big_contour = []
        max = 0
        for i in contours:
            area = cv2.contourArea(i)  # --- find the contour having biggest area ---
            if (area > 5000):
                max = area
                big_contour.append(i)

        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)

        # draw rectangles around contours
        for contour in big_contour:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.drawContours(self.img, big_contour, -1, (255, 0, 0), 3)

        cv2.imshow('i', self.img)
        cv2.waitKey(0)

a = SelectNucleus()
a.split_nucleus()
