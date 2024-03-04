from eduset.utils.camera import grab_pic, save_img
from IPython.display import clear_output

import matplotlib.pyplot as plt
import numpy as np
import pypylon.pylon as pylon

import time


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
