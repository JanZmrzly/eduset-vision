from eduset.camera import grab_pic, save_img
from IPython.display import clear_output
from ultralytics.utils.plotting import Annotator

import matplotlib.pyplot as plt
import numpy as np
import pypylon.pylon as pylon

import time
import cv2


def collect_dataset(cam: pylon.InstantCamera, path: str, count: int, period: int):
    for i in range(0, count):
        # Start counter
        print("Countdown has begun")
        for j in range(period-1, -1, -1):
            print(f"{j} seconds left")
            time.sleep(1)

        print("Grabbing picture...")
        img = grab_pic(cam)

        try:
            save_img(img, str(i), path)

            img_array = img.GetArray()
            negative_img = np.max(img_array) - img_array
            plt.imshow(negative_img, cmap="gray")
            plt.show()
        except pylon.InvalidArgumentException:
            print("Image was NOT saved")

        time.sleep(1)
        clear_output()

    print(f"Dataset was created successfully - path:\t{path}")


def visualize_bbox(img, bbox, class_name, color, thickness=2):
    # source: https://albumentations.ai/docs/examples/example_bboxes/

    """ Visualizes a single bounding box on the image   """

    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=(255, 255, 255),
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    # source: https://albumentations.ai/docs/examples/example_bboxes/

    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name, color=(255, 0, 0))
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)


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


def disp_masks(img_path: str, annotations_path: str, classes: dict):
    img = cv2.imread(img_path)
    annotator = Annotator(img)

    annotations = load_annotations(annotations_path)
    for annt in annotations:
        clss, mask = create_mask(annt)
        annotator.seg_bbox(mask,
                           det_label=classes[clss].get("name"),
                           mask_color=classes[clss].get("color"))
    
    result = annotator.result()
    plt.imshow(result)

