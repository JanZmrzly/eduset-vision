from vision import EdusetVision
from eduset.utils.logger import EdusetLogger

from fastapi import FastAPI, responses

import uvicorn
import tempfile
import cv2 as cv

app = FastAPI()
logger = EdusetLogger("API")
vision = EdusetVision()


@app.get("/placement")
async def read_placement(image: bool = False):
    vision.update_placement()
    placement = vision.placement
    vision.disp_placement() if vision.verbose else None

    if image:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = vision.predicted_image
            cv.imwrite(tmp.name, img)
            logger.info("Predicted image sent")
            return responses.FileResponse(tmp.name, media_type="image/png")

    return responses.JSONResponse(content=placement)


@app.get("/disconnect")
async def disconnect():
    vision.shut_down()
    logger.warning("Camera disconnected")

    return "Disconnected"


@app.get("/connect")
async def connect(ip: str = "emulation"):
    vision.connect(ip)

    return "Connected"


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
