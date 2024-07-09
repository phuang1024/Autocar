"""
Scripts for generating train data.
"""

import time
from pathlib import Path

import cv2

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

        while True:
            img_rgb = read_latest(q_rgb).getCvFrame()

            if interface.rc_values[5] > 0.5:
                cv2.imwrite(str(dir / f"{i}.jpg"), img_rgb)
                with open(dir / f"{i}.txt", "w") as f:
                    f.write(f"{interface.rc_values[0]}\n")
                print("Write", i)
                i += 1

            time.sleep(args.interval)
