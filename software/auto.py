import time
from math import atan2
from threading import Thread

import depthai
import matplotlib.pyplot as plt
import numpy as np
import torch

from camera import *
from train import AutocarModel, DEVICE


class Plotter:
    """
    Multi line online plotter.
    """

    def __init__(self, lines, max_len=100):
        """
        lines: List of str, names of the lines.
        """
        self.lines = lines
        self.max_len = max_len
        self.data = np.zeros((len(lines), max_len))

    def update(self, values):
        """
        Plot in matplotlib.

        values: List of float, values to plot.
        """
        self.data = np.roll(self.data, -1, axis=1)
        self.data[:, -1] = values

        plt.clf()
        for i, line in enumerate(self.lines):
            plt.plot(self.data[i], label=line)
        plt.legend()
        plt.show()


class Derivative:
    """
    Calculate derivative via streaming data and EMA.
    """

    def __init__(self, k=0.4):
        self.k = k
        self.value = 0
        self.last_value = 0
        self.last_time = time.time()

    def update(self, value):
        """
        Computes velocity = delta(value) / delta(time).
        """
        now = time.time()
        dt = now - self.last_time
        dt = max(dt, 1e-3)
        d_angle = value - self.last_value

        vel = d_angle / dt
        self.last_value = value
        self.last_time = now

        self.value = self.k * vel + (1 - self.k) * self.value

        return self.value


def rc_ctrl_loop(args):
    """
    Converts steering input into speed and steer values.
    Reduces speed when steering.
    Uses IMU to monitor turning speed.

    Will modify `interface.steer_input` based on steer.

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

    deriv_func = Derivative()
    integral = 0
    """PID integral term."""
    target = 0
    """Target value (angular position)."""
    kp = 1
    ki = 0.5
    kd = 0.3

    last_angle = 0
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

        # Compute euler Z angle
        forward = rot @ np.array([0, 1, 0])
        angle = atan2(forward[0], forward[1])
        if abs(angle - last_angle) > 5:
            # Handle 2pi wrap
            target = angle
        last_angle = angle

        target = 0.2 * (args["steer"] * 0.5 + angle) + 0.8 * target
        error = target - angle
        deriv = deriv_func.update(error)
        integral += error
        integral *= 0.8

        steer = kp * error + ki * integral + kd * deriv
        interface.steer_input = steer


@torch.no_grad()
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

        ctrl = 0
        while True:
            images = wrapper.get()

            nn_enabled = interface.rc_values[5] > 0.5

            # Infer model
            if nn_enabled:
                if is_onnx:
                    y = images["nn"].getData().view(np.float16).item()
                else:
                    x = images_to_tensor(images)
                    x = x.float() / 255
                    x = x.unsqueeze(0).to(DEVICE)
                    y = model(x).item()

                ctrl = 0.5 * ctrl + 0.5 * y
                print("Model output:", y, "Control:", ctrl)

            else:
                ctrl = 0

            ctrl_args["steer"] = ctrl

            time.sleep(0.01)
