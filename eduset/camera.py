import time
import pypylon.pylon as pylon
# import logging


def get_devices():
    tlf = pylon.TlFactory.GetInstance()
    devices = tlf.EnumerateDevices()

    print("Model name\t\tSerial number")
    for device in devices:
        print(device.GetModelName() + "\t" + device.GetSerialNumber())


def reset(cam: pylon.InstantCamera):
    cam.UserSetSelector.SetValue = "Default"
    cam.UserSetLoad.Execute()
    print("Camera was set to Default")


def get_info(cam: pylon.InstantCamera):
    print(f"Trigger Selector: {cam.TriggerSelector.Symbolics}")
    print(f"Pixel Format: {cam.PixelFormat.Symbolics}")
    print(f"Gain: {cam.Gain.Value}")


def connect(device: int) -> pylon.InstantCamera:
    tlf = pylon.TlFactory.GetInstance()
    devices = tlf.EnumerateDevices()
    cam = pylon.InstantCamera(tlf.CreateDevice(devices[device]))
    cam.Open()
    print(f"Successfully connected to {devices[device].GetModelName()}")

    return cam


def disconnect(cam: pylon.InstantCamera):
    cam.Close()
    print(f"Successfully disconnected")


def preview_continuous(cam: pylon.InstantCamera):
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
            print("Error: ", grab_result.ErrorCode)
        grab_result.Release()
        time.sleep(0.01)

        if not image_window.IsVisible():
            cam.StopGrabbing()

    image_window.Close()


def grab_pic(cam: pylon.InstantCamera) -> pylon.PylonImage:
    pic = cam.GrabOne(100)
    img = pylon.PylonImage()
    img.Release()
    img.AttachGrabResultBuffer(pic)

    return img


def save_img(img: pylon.PylonImage, name: str, path: str):
    location = f"{path}/{name}.png"
    img.Save(pylon.ImageFileFormat_Png, location)
    print(f"Image {name} was saved to {path}")
