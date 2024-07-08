"""
Scripts for generating train data.
"""

import time
from math import atan2
from pathlib import Path

import cv2

from camera import *


def get_z_euler(quat):
    euler_z = atan2(2 * (quat.real * quat.i + quat.j * quat.k), 1 - 2 * (quat.i ** 2 + quat.j ** 2))
    return euler_z


def gen_data(args, interface):
    dir = Path(args.dir)
    dir.mkdir(exist_ok=True, parents=True)

    i = 0
    for file in dir.iterdir():
        if file.is_file and file.stem.isdigit():
            i = max(i, int(file.stem) + 1)

    interface.add_thread(interface.standard_rc)

    pipeline = create_pipeline(args.res)
    print("Setup Depthai pipeline.")
    with depthai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")
        q_imu = device.getOutputQueue("imu")

        while True:
            img_rgb = read_latest(q_rgb).getCvFrame()

            imu_data = read_latest(q_imu)
            quat = imu_data.packets[0].rotationVector
            euler_z = get_z_euler(quat)

            """
            cv2.imshow("rgb", img_rgb)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            """

            if interface.rc_values[5] > 0.5:
                cv2.imwrite(str(dir / f"{i}.jpg"), img_rgb)
                with open(dir / f"{i}.txt", "w") as f:
                    f.write(f"{euler_z}\n{quat.real}\n{quat.i}\n{quat.j}\n{quat.k}\n")
                print("Write", i)
                i += 1

            time.sleep(args.interval)
