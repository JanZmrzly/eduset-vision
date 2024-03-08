import os
import logging
import uvicorn
import asyncio

import pypylon.pylon as pylon
import cv2 as cv
import numpy as np

from eduset.utils.camera import get_devices, connect, disconnect, grab_pic, custom_emulation, update_emulation
from eduset.utils.placement import get_placement
from datetime import datetime
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from asyncio import LifoQueue

# FIXME: add api clearer way
_app = FastAPI()
_shared_queue = LifoQueue(maxsize=1)

# TODO: add logger
_logger = logging.getLogger(__name__)


@_app.get("/placement")
async def number() -> JSONResponse:
    placement = await _shared_queue.get()
    return JSONResponse(content=placement)


def get_time() -> str:
    current_time = datetime.now()

    year = current_time.year
    month = current_time.month
    day = current_time.day
    hour = current_time.hour
    minute = current_time.minute
    second = current_time.second

    return f"{day}-{month}-{year} {hour}:{minute}.{second}"


def _get_placement(cam: pylon.InstantCamera, model: YOLO, classes: dict | None,
                   model_confidence=0.8, visualisation=False) -> dict | None:

    placement = None
    try:
        pylon_img = grab_pic(cam)
        img = pylon_img.GetArray()
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

        results = model.predict(source=img, conf=model_confidence, verbose=False)
        results = [result.cpu() for result in results]

        placement, predicted_image = get_placement(img, results, classes)

        if visualisation is True:
            cv.imshow(winname="Prediction&Placement", mat=predicted_image)
            cv.waitKey(1)
    except ValueError:
        print("Bad camera connection")

    return placement


def disp_placement(placement: dict) -> None:
    print(f"[{get_time()}]")
    [print(placement[key]) for key in placement]


async def vision_routine() -> None:
    await asyncio.sleep(2)  # Starting API server

    os.environ["PYLON_CAMEMU"] = "2"
    model_path = "../samples/yolo_runs/model/model_v12/weights/best.pt"
    model_confidence = 0.8
    verbose = True
    visualization = True
    classes = {
                    "0": {"name": "circle", "color": [255, 0, 0], "symmetrical": True},
                    "1": {"name": "square", "color": [0, 255, 0], "symmetrical": True},
                    "2": {"name": "triangle", "color": [0, 0, 255], "symmetrical": False}
              }
    width = 1100
    height = 1100
    emulation = True
    fail_emulation = False
    orig_shape = (2448, 2048)
    image_file_name = "/mnt/c/Users/zmrzl/OneDrive/Documents/PROJEKTY/005_DIPLOMOVA_PRACE/003_DATASET/DS/DS12/"

    logging.getLogger("eduset_vision").setLevel(logging.INFO)

    get_devices()
    cam = connect(0)

    model = YOLO(model_path)

    if emulation is True:
        custom_emulation(cam, image_file_name, fail=fail_emulation, orig_shape=orig_shape)

    if visualization is True:
        cv.namedWindow(winname="Prediction&Placement", flags=cv.WINDOW_GUI_NORMAL)
        cv.resizeWindow(winname="Prediction&Placement", width=width, height=height)
        img = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        cv.imshow(winname="Prediction&Placement", mat=img)
        cv.waitKey(1)

    try:
        while True:
            placement = _get_placement(cam, model, classes, model_confidence, visualization)
            await _shared_queue.put(placement)
            await asyncio.sleep(1)
            if not _shared_queue.empty():
                _shared_queue.get_nowait()
                _shared_queue.task_done()

            if verbose is True:
                disp_placement(placement)

            if emulation is True:
                update_emulation(cam, image_file_name)

    finally:
        print("\nThe program was terminated by pressing a key...")
        disconnect(cam)
        cv.destroyAllWindows()


async def main():
    uvicorn_config = uvicorn.Config(_app, host="localhost", port=8000, loop="asyncio")
    server = uvicorn.Server(uvicorn_config)
    loop = asyncio.get_event_loop()

    loop.create_task(vision_routine())
    await server.serve()
