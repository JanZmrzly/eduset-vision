import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from typing import List


def plot_results(paths: List[str], figsize=(15, 5)):
    fig, axes = plt.subplots(nrows=1, ncols=len(paths), figsize=figsize)

    for ax, image_path in zip(axes, paths):
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.set_xticks([]), ax.set_yticks([])


def draw_axes(image: np.ndarray) -> np.ndarray:
    marker = cv.imread("../eduset/utils/marker.png")
    h, w = marker.shape[:2]

    overlay_img = np.ones(image.shape, np.uint8) * 255
    overlay_img[0:h, 0:w] = marker
    gray_marker = cv.cvtColor(overlay_img, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray_marker, thresh=240, maxval=255, type=cv.THRESH_BINARY_INV)
    mask_inv = cv.bitwise_not(mask)

    temp1 = cv2.bitwise_and(image, image, mask=mask_inv)
    temp2 = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)

    image = cv.add(temp1, temp2)
    return image
