import random
import time

import glob as glob
import pypylon.pylon as pylon

from eduset.utils.logger import EdusetLogger

logger = EdusetLogger(__name__)


def get_devices() -> None:
    tlf = pylon.TlFactory.GetInstance()
    devices = tlf.EnumerateDevices()

    logger.info("Model name\t\tSerial number")
    for device in devices:
        logger.info(device.GetModelName() + "\t" + device.GetSerialNumber())


def reset(cam: pylon.InstantCamera) -> None:
    cam.UserSetSelector.SetValue = "Default"
    cam.UserSetLoad.Execute()
    logger.info("Camera was set to Default")


def get_info(cam: pylon.InstantCamera) -> None:
    logger.info(f"Trigger Selector: {cam.TriggerSelector.Symbolics}")
    logger.info(f"Pixel Format: {cam.PixelFormat.Symbolics}")
    logger.info(f"Gain: {cam.Gain.Value}")


def connect(device: int) -> pylon.InstantCamera:
    tlf = pylon.TlFactory.GetInstance()
    devices = tlf.EnumerateDevices()
    cam = pylon.InstantCamera(tlf.CreateDevice(devices[device]))
    cam.Open()
    logger.info(f"Successfully connected to {devices[device].GetModelName()}")

    return cam


def disconnect(cam: pylon.InstantCamera) -> None:
    cam.Close()
    logger.info(f"Successfully disconnected")


# FIXME: Only for Windows
def preview_continuous(cam: pylon.InstantCamera) -> None:
    image_window = pylon.PylonImageWindow()
    image_window.Create(1)

    cam.StartGrabbing(10000, pylon.GrabStrategy_LatestImageOnly)
    while cam.IsGrabbing():
        # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        grab_result = cam.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

        # Image grabbed successfully?
        if grab_result.GrabSucceeded():
            image_window.SetImage(grab_result)
            image_window.Show()
        else:
            # grabResult.ErrorDescription does not work properly in python could throw UnicodeDecodeError
            logger.error(grab_result.ErrorCode)
        grab_result.Release()
        time.sleep(0.01)

        if not image_window.IsVisible():
            cam.StopGrabbing()

    image_window.Close()


def grab_pic(cam: pylon.InstantCamera) -> pylon.PylonImage:
    pic = cam.GrabOne(1000)
    img = pylon.PylonImage()
    img.Release()
    img.AttachGrabResultBuffer(pic)

    return img


def save_img(img: pylon.PylonImage, name: str, path: str) -> None:
    location = f"{path}/{name}_{round(time.time())}.png"
    img.Save(pylon.ImageFileFormat_Png, location)
    logger.info(f"Image {name} was saved to {path}")


def custom_emulation(cam: pylon.InstantCamera, image_file_name: str, orig_shape: tuple, fail=True) -> None:
    cam.TestImageSelector = "Off"
    cam.ImageFileMode = "On"
    cam.ImageFilename = image_file_name
    cam.Width = orig_shape[0]
    cam.Height = orig_shape[1]

    if fail is True:
        cam.ForceFailedBufferCount.Value = 10
        cam.ForceFailedBuffer.Execute()


def update_emulation(cam: pylon.InstantCamera, images_dir: str) -> None:
    test_images = glob.glob(f"{images_dir}*.png")
    img_path = random.choice(test_images)
    cam.ImageFilename = img_path
