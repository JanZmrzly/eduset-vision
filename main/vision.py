import os

import pypylon.pylon as pylon
import cv2 as cv
import numpy as np

from eduset.utils.camera import connect, disconnect, grab_pic, custom_emulation, update_emulation
from eduset.utils.placement import get_placement
from eduset.utils.plots import draw_axes
from eduset.utils.logger import EdusetLogger

from ultralytics import YOLO


logger = EdusetLogger(__name__)


class EdusetVision:
    def __init__(self):
        os.environ["PYLON_CAMEMU"] = "2"

        self.model_path = "/mnt/c/Users/zmrzl/OneDrive/Desktop/yolo_runs/model/model_v15/weights/best.pt"
        self.model_confidence = 0.8
        self.verbose = True
        self.visualization = False
        self.classes: dict | None = {
            "0": {"name": "circle", "color": [255, 0, 0], "symmetrical": True},
            "1": {"name": "square", "color": [0, 255, 0], "symmetrical": True},
            "2": {"name": "triangle", "color": [0, 0, 255], "symmetrical": False}
        }
        self.width = 640
        self.height = 640
        self.emulation = True
        self.fail_emulation = False
        self.orig_shape = (2448, 2048)
        self.image_file_name = "/mnt/c/Users/zmrzl/OneDrive/Documents/PROJEKTY/005_DIPLOMOVA_PRACE/003_DATASET/DS/DS6/"

        self.cam: pylon.InstantCamera | None = connect(0)
        self.model = YOLO(self.model_path)

        self.placement: dict | None = None
        self.predicted_image: np.ndarray | None = None

        if self.emulation is True:
            custom_emulation(self.cam, self.image_file_name, fail=self.fail_emulation, orig_shape=self.orig_shape)

        if self.visualization is True:
            cv.namedWindow(winname="Prediction&Placement", flags=cv.WINDOW_GUI_NORMAL)
            cv.resizeWindow(winname="Prediction&Placement", width=self.width, height=self.height)
            img = np.zeros(shape=(self.height, self.width, 3), dtype=np.uint8)
            cv.imshow(winname="Prediction&Placement", mat=img)
            cv.waitKey(1)

    def update_placement(self, real_shape=np.array([314, 386])) -> None:
        self.placement = None
        self.predicted_image = None

        if self.cam is None:
            logger.warning("Camera is NOT connected")
            return None

        if self.emulation is True:
            update_emulation(self.cam, self.image_file_name)

        try:
            logger.info("Getting placement")
            pylon_img = grab_pic(self.cam)
            img = pylon_img.GetArray()
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

            results = self.model.predict(source=img, conf=self.model_confidence, verbose=False)
            results = [result.cpu() for result in results]

            self.placement, self.predicted_image = get_placement(img, results, self.classes, real_shape)
            self.predicted_image = draw_axes(self.predicted_image)

            if self.visualization:
                cv.imshow(winname="Prediction&Placement", mat=self.predicted_image)
                cv.waitKey(1)

            logger.info("Placement found") if self.placement else logger.warning("Placement NOT found")
        except ValueError:
            logger.warning("Bad camera connection")

    def disp_placement(self) -> None:
        if self.placement:
            [print(self.placement[key]) for key in self.placement]

    def shut_down(self) -> None:
        logger.info("The program was terminated")
        disconnect(self.cam)
        self.cam = None
        cv.destroyAllWindows()

    def connect(self, ip: str) -> None:
        # TODO: add connect with IP
        self.cam = connect(0)
