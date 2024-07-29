import time
from threading import Thread

import depthai
import numpy as np
import torch

from camera import *
from train import AutocarModel, DEVICE


def rc_ctrl_loop(args):
    """
    Converts steering input into speed and steer values.
    Reduces speed when steering.
    Uses IMU to monitor turning speed.

    args: Dict.
        interface: Interface instance.
        device: Depthai device.
        steer: Steering value.
        run: Run flag.
    """
    interface = args["interface"]
    device = args["device"]

    q_imu = device.getOutputQueue("imu")
    q_imu.setMaxSize(1)
    q_imu.setBlocking(False)

    while args["run"]:
        # Get rotation matrix
        imu_data = q_imu.get().packets[0].rotationVector
        imu_data = np.array([imu_data.real, imu_data.i, imu_data.j, imu_data.k])
        imu_data = imu_data / np.linalg.norm(imu_data)
        w, x, y, z = imu_data
        rot = np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)],
        ])

        print(rot)


def auto_main(args, interface):
    """
    Main for auto driving.
    """
    is_onnx = args.model_path.suffix == ".blob"

    if is_onnx:
        print("Load ONNX model:", args.model_path)
        pipeline = create_pipeline(args.res, args.model_path)
    else:
        model = AutocarModel().to(DEVICE)
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        print("Load Pytorch model:", args.model_path)
        pipeline = create_pipeline(args.res)

    with depthai.Device(pipeline) as device:
        interface.add_thread(interface.auto_rc)

        ctrl_args = {
            "interface": interface,
            "device": device,
            "steer": 0,
            "run": True,
        }
        ctrl_thread = Thread(target=rc_ctrl_loop, args=(ctrl_args,))
        ctrl_thread.start()

        wrapper = PipelineWrapper(device, include_nn=is_onnx)

        while True:
            images = wrapper.get()

            nn_enabled = interface.rc_values[5] > 0.5

            # Infer model
            if nn_enabled:
                if is_onnx:
                    pred = images["nn"].getData().view(np.float16).item()
                else:
                    with torch.no_grad():
                        x = images_to_tensor(images)
                        x = x.float() / 255
                        x = x.unsqueeze(0).to(DEVICE)
                        pred = model(x).item()

                interface.speed_mult = 1 - abs(pred) / 2

                print("Pred", pred)

            else:
                pred = 0
                interface.speed_mult = 1

            interface.nn_pred = pred

            time.sleep(0.01)
