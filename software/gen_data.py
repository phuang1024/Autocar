"""
Scripts for generating train data.
"""

import time
from pathlib import Path

import cv2

from camera import *


class DataGen:
    def __init__(self, dir):
        self.dir = dir
        self.i = 0
        for file in dir.iterdir():
            if file.is_file and file.stem.isdigit():
                self.i = max(self.i, int(file.stem) + 1)

    def write(self, images, label):
        cv2.imwrite(str(self.dir / f"{self.i}.rgb.jpg"), images["rgb"])
        cv2.imwrite(str(self.dir / f"{self.i}.depth.jpg"), images["depth"])
        cv2.imwrite(str(self.dir / f"{self.i}.depth_conf.jpg"), images["depth_conf"])

        with open(self.dir / f"{self.i}.txt", "w") as f:
            f.write(f"{label}\n")

        print("Write", self.i)
        self.i += 1


def gen_data_main(args, interface):
    dir = Path(args.dir)
    dir.mkdir(exist_ok=True, parents=True)
    data_gen = DataGen(dir)

    if not args.self_rc:
        interface.add_thread(interface.auto_rc)

    pipeline = create_pipeline(args.res)
    print("Setup Depthai pipeline.")
    with depthai.Device(pipeline) as device:
        wrapper = PipelineWrapper(device)

        while True:
            images = wrapper.get()

            if args.self_rc:
                cv2.imshow("asdf", images["depth_fac"])
                cv2.waitKey(100)

            if interface.rc_values[5] > 0.5:
                data_gen.write(images, 1)

            time.sleep(0.01)
