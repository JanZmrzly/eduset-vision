import torch
import os

import cv2 as cv
import numpy as np

from eduset.fastercnn.predict import Predictor
from tqdm import tqdm
from typing import List


class VideoPredictor(Predictor):
    def __init__(self, input_path: str, output_path: str, num_classes: int, classes: List[str],
                 device=None, size=640, threshold=0.6
                 ) -> None:
        super(VideoPredictor, self).__init__(
            input_path=input_path,
            output_path=output_path,
            num_classes=num_classes,
            classes=classes,
            device=device,
            size=size,
            threshold=threshold
        )

    def video_predict(self, classes_vis: dict | None, plot=True, save=True) -> None:
        cap = cv.VideoCapture(self.input_path)
        name = os.path.basename(self.input_path)
        name, _ = os.path.splitext(name)
        output_path = f"{self.output_path}/{name}.mp4"
        if not cap.isOpened():
            print("Cannot open video")
            exit(ImportError)
        print(f"Video to test: {name} loaded SUCCESSFULLY")

        os.makedirs(name=f"{self.output_path}", exist_ok=True)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        if classes_vis is not None:
            colors = [class_info["color"] for class_info in classes_vis.values()]
            self.colors = np.array(colors, dtype=np.int32)

        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        progress_bar = tqdm(total=total_frames, desc="Processing video frames", unit="frame")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            orig_frame = frame.copy()

            frame_input = self.preprocess(frame)

            with torch.no_grad():
                prediction = self.model.model(frame_input.to(self.device))

            # Load all detection to CPU for further operations
            prediction = [{k: v.to('cpu') for k, v in t.items()} for t in prediction]

            bboxes, bscores, image_classes = self.prepare_bboxes(prediction)
            predicted_frame = self.draw_bboxes(bboxes, bscores, image_classes, orig_frame)

            out.write(predicted_frame) if save is True else print("Saving is disabled")
            progress_bar.update(1)

        progress_bar.close()
        cap.release()
        out.release()
