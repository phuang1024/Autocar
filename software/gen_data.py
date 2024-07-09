"""
Scripts for generating train data.
"""

import time
from pathlib import Path

import cv2
import numpy as np

from camera import *


def gen_data(args, interface):
    dir = Path(args.dir)
    dir.mkdir(exist_ok=True, parents=True)

    i = 0
    for file in dir.iterdir():
        if file.is_file and file.stem.isdigit():
            i = max(i, int(file.stem) + 1)

    interface.add_thread(interface.auto_rc)

    pipeline = create_pipeline(args.res)
    print("Setup Depthai pipeline.")
    with depthai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")
        q_depth = device.getOutputQueue("depth")
        q_depth_conf = device.getOutputQueue("depth_conf")

        while True:
            img_rgb = read_latest(q_rgb).getCvFrame()
            img_depth = read_latest(q_depth).getFrame()
            img_depth = (img_depth / np.max(img_depth) * 255).astype(np.uint8)
            img_depth_conf = read_latest(q_depth_conf).getFrame()

            cv2.imshow("depth", img_depth)
            cv2.imshow("depth_conf", img_depth_conf)
            cv2.waitKey(1)

            if interface.rc_values[5] > 0.5:
                cv2.imwrite(str(dir / f"{i}.jpg"), img_rgb)

                label = interface.rc_values[0] * 2 - 1
                with open(dir / f"{i}.txt", "w") as f:
                    f.write(f"{label}\n")

                print("Write", i)
                i += 1

            time.sleep(args.interval)
