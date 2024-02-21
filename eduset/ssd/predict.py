# source: https://debuggercafe.com/train-ssd300-vgg16/
import torch
import os

import cv2 as cv
import glob as glob
import numpy as np
import matplotlib.pyplot as plt

from eduset.ssd.ssd import Model
from typing import List


class Predictor:
    def __init__(self, input_path: str, output_path: str, num_classes: int, classes: List[str],
                 device=None, size=640, threshold=0.6) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.size = size
        self.threshold = threshold
        self.num_classes = num_classes
        self.device = device
        self.classes = classes

        self.model = self.get_model()
        self.colors = np.random.uniform(low=0, high=255, size=(self.num_classes, 3))

    def get_model(self) -> Model:
        new_model = Model(epochs=0, train_dataloader=None, val_dataloader=None, out_dir=self.output_path)
        return new_model

    def load(self, model_path: str) -> None:
        self.model.create(self.num_classes, self.size)
        milestone = torch.load(f=model_path, map_location=self.device)
        self.model.model.load_state_dict(milestone["model_state_dict"])
        self.model.switch_gpu()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.model.eval()

    def predict(self, classes_vis: dict | None, plot=True, save=True) -> None:
        # Load images
        images = glob.glob(f"{self.input_path}/*.jpg")
        print(f"Images to test: {len(images)} loaded SUCCESSFULLY")

        if classes_vis is not None:
            colors = [class_info["color"] for class_info in classes_vis.values()]
            self.colors = np.array(colors, dtype=np.int32)

        for im, image_path in enumerate(images):
            image_name = os.path.basename(image_path)
            image = cv.imread(image_path)
            orig_image = image.copy()
            print("\n")
            print(f"Image #{im}\tValidating image {image_name}")

            image_input = self.preprocess(image)

            with torch.no_grad():
                prediction = self.model.model(image_input.to(self.device))

            # Load all detection to CPU for further operations
            prediction = [{k: v.to('cpu') for k, v in t.items()} for t in prediction]

            bboxes, bscores, image_classes = self.prepare_bboxes(prediction)
            predicted_image = self.draw_bboxes(bboxes, bscores, image_classes, orig_image)

            self.disp_image(predicted_image) if plot is True else print("Plot is disabled")
            self.save_image(predicted_image, image_name) if save is True else print("Saving is disabled")

    def preprocess(self, image: np.ndarray) -> torch.tensor:
        image = cv.resize(image, dsize=(self.size, self.size))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32)
        image /= 255

        # Transpose height, weight, color -> color, height, weight
        image_input = np.transpose(image, axes=(2, 0, 1)).astype(np.float32)
        image_input = torch.tensor(image_input, dtype=torch.float).cuda()
        # Add batch dimension
        image_input = torch.unsqueeze(image_input, dim=0)

        return image_input

    def save_image(self, image: np.ndarray, image_name: str) -> None:
        os.makedirs(name=f"{self.output_path}", exist_ok=True)
        cv.imwrite(filename=f"{self.output_path}/{image_name}", img=image)
        print(f"{image_name} was saved to {self.output_path}")

    @staticmethod
    def disp_image(image: np.ndarray) -> None:
        plt.imshow(image)
        plt.show()

    def draw_bboxes(self, bboxes: np.array, bscores: np.array, image_classes: np.array,
                    orig_image: np.ndarray) -> np.ndarray:

        for j, box in enumerate(bboxes):
            name = image_classes[j]
            score = bscores[j]
            color = self.colors[self.classes.index(name)]
            x_min = int((box[0] / self.size) * orig_image.shape[1])
            y_min = int((box[1] / self.size) * orig_image.shape[0])
            x_max = int((box[2] / self.size) * orig_image.shape[1])
            y_max = int((box[3] / self.size) * orig_image.shape[0])

            cv.rectangle(
                orig_image,
                (x_min, y_min), (x_max, y_max),
                color=(int(color[0]), int(color[1]), int(color[2])),
                thickness=2
            )
            cv.putText(
                orig_image,
                text=f"{name} {score:.2f}",
                org=(x_min, y_min-10),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(int(color[0]), int(color[1]), int(color[2])),
                thickness=5
            )

        predicted_image = orig_image

        return predicted_image

    def prepare_bboxes(self, predictions: torch.tensor) -> [np.array, np.array, np. array]:
        boxes = predictions[0]["boxes"].data.numpy()
        scores = predictions[0]["scores"].data.numpy()

        # Select only boxes with score above threshold
        boxes = boxes[scores >= self.threshold].astype(np.int32)
        scores = scores[scores >= self.threshold]
        bboxes = boxes.copy()
        bscores = scores.copy()
        image_classes = [self.classes[i] for i in predictions[0]['labels'].cpu().numpy()]

        return bboxes, bscores, image_classes
