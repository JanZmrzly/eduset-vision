import os

import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt

from ultralytics.utils.plotting import Annotator
from IPython.display import clear_output
from scipy.signal import savgol_filter


def create_mask(annotation: np.array, img_shape=np.array([640, 640])) -> (int, np.array):
    clss = int(annotation[0])

    mask = annotation[1:]
    mask = np.reshape(mask, newshape=(np.shape(mask)[0] // 2, 2))
    mask = mask * img_shape

    return clss, mask


def load_annotations(path: str) -> list[list[float]]:
    annotations = []

    with open(path, 'r') as file:
        for line in file:
            values = [float(value) for value in line.strip().split()]
            annotations.append(values)

    return annotations


def disp_masks(img_path: str, annotations_path: str, classes: dict) -> None:
    img = cv.imread(img_path)
    annotator = Annotator(img)

    annotations = load_annotations(annotations_path)
    for annt in annotations:
        clss, mask = create_mask(annt)
        annotator.seg_bbox(mask,
                           det_label=classes[str(clss)].get("name"),
                           mask_color=classes[str(clss)].get("color"))

    result = annotator.result()
    plt.imshow(result)


def download_model(out_dir: str, model: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    url = f"https://github.com/ultralytics/assets/releases/download/v8.1.0/{model}"
    out_dir = f"{out_dir}/{model}"

    os.system(f"wget {url} -O {out_dir}")
    clear_output()
    print(f"Model {model}: Downloaded successfully to {out_dir}")


def save_results(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(name=out_dir, exist_ok=True)
    loss_score = df.iloc[:, 1].values
    map_score = df.iloc[:, 8].values
    map50_score = df.iloc[:, 7].values

    graphs = {"loss score": loss_score, "mAP": map_score, "mAP50": map50_score}
    for key, value in graphs.items():
        x = np.arange(1, len(value) + 1)
        plt.scatter(x, value)
        plt.plot(x, value, label="results")

        if len(value) >= 20:
            smoothed = np.convolve(value, np.ones(2) / 2, mode='valid')
            smoothed = np.insert(smoothed, 0, value[0])
            smoothed = np.append(smoothed, value[-1])
            xx = np.arange(1, len(smoothed) + 1)
            plt.plot(xx, smoothed, "--", color="orange", label="smooth")

        plt.ylabel(key)
        plt.xlabel("epoch")
        plt.legend()

        plt.savefig(f"{out_dir}/{key}.png", format="png")
        plt.clf()

    print(f"Metrics have been saved to {out_dir}")


def smooth(mask: np.ndarray) -> np.ndarray:
    smooth_mask = np.apply_along_axis(lambda x: savgol_filter(x, window_length=5, polyorder=3), axis=0, arr=mask)
    return smooth_mask
