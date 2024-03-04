import cv2 as cv
import numpy as np

from ultralytics.engine.results import Results
from ultralytics.utils.plotting import Annotator
from eduset.yolo.yolo import smooth
from typing import List


def get_placement(orig_image: np.ndarray, results: List[Results], classes: dict) -> (dict, np.ndarray):
    annotator = Annotator(orig_image.copy())
    placement0 = {}
    placement1 = {}

    for result in results:
        for i, box in enumerate(result.boxes):
            label = classes[str(int(box.cls.item()))]["name"]
            conf = box.conf.item()
            color = classes[str(int(box.cls.item()))]["color"]
            symmetrical = classes[str(int(box.cls.item()))]["symmetrical"]
            box_shape = box.xyxy.view(-1)
            mask = result.masks.xy[i]

            # mask_image = get_contour(orig_image, mask)
            # clipping = get_clipping(mask_image, box_shape.numpy())
            mask = smooth(mask)

            # center, angle = pca_get_angle(mask)
            # center1, angle1 = symmetric_get_angle(mask)

            center, angle = symmetric_get_angle(mask) if symmetrical else pca_get_angle(mask)

            placement0[len(placement0)] = {"name": label, "x": center[0], "y": center[1], "angle": angle}
            # placement1[len(placement1)] = {"name": label, "x": center1[0], "y": center1[1], "angle": angle1}

            annotator.box_label(box_shape, label=f"{label} | {conf:.2f}", color=color)

    image = annotator.result()
    image = draw_rotations(image, placement0, color=(255, 255, 255))
    # image = draw_rotations(image, placement1, color=(255, 255, 0))

    return placement0, image


def draw_rotations(image: np.ndarray, placement: dict, color=(0, 0, 0)) -> np.ndarray:
    for key, value in placement.items():
        center = (value["x"], value["y"])
        delta_x = 0.08 * image.shape[0] * np.cos(value["angle"])
        delta_y = 0.08 * image.shape[0] * np.sin(value["angle"])
        end_point = (center[0] + int(delta_x), center[1] + int(delta_y))
        cv.arrowedLine(image, center, end_point, color=color, thickness=10, tipLength=0.1)
        cv.circle(image, center=center, radius=10, color=color, thickness=-1)
    return image


def get_contour(original_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img_contours = np.zeros_like(original_image)
    cv.polylines(img_contours, pts=[np.int32([mask])], isClosed=True, color=(255, 255, 255), thickness=2)
    return img_contours


def get_clipping(original_image: np.ndarray, box_shape: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = int(box_shape[0]), int(box_shape[1]), int(box_shape[2]), int(box_shape[3])
    clipping = original_image[y0: y1, x0: x1]
    return clipping


def pca_get_angle(points: np.ndarray) -> (tuple, float):
    data_pts = points

    # Perform PCA analysis
    mean = np.empty(0)
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

    # Store the center of the object
    center = (int(mean[0, 0]), int(mean[0, 1]))
    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])

    return center, angle


def symmetric_get_angle(points: np.ndarray) -> (tuple, float):
    box = cv.minAreaRect(points)
    center = (int(box[0][0]), int(box[0][1]))
    angle = (box[2] * np.pi) / 180.0
    return center, angle

