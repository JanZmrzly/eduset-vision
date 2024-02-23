import cv2 as cv
import matplotlib.pyplot as plt

from typing import List


def plot_results(paths: List[str], figsize=(15, 5)):
    fig, axes = plt.subplots(nrows=1, ncols=len(paths), figsize=figsize)

    for ax, image_path in zip(axes, paths):
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.set_xticks([]), ax.set_yticks([])
