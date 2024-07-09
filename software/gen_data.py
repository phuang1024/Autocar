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
        wrapper = PipelineWrapper(device)

        while True:
            images = wrapper.get()

            if interface.rc_values[5] > 0.5:
                cv2.imwrite(str(dir / f"{i}.rgb.jpg"), images["rgb"])
                cv2.imwrite(str(dir / f"{i}.depth.jpg"), images["depth"])
                cv2.imwrite(str(dir / f"{i}.depth_conf.jpg"), images["depth_conf"])

                label = interface.rc_values[0] * 2 - 1
                with open(dir / f"{i}.txt", "w") as f:
                    f.write(f"{label}\n")

                print("Write", i)
                i += 1

            time.sleep(args.interval)
