import math
import time

import torch
import torchvision

from camera import *


def create_model():
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(512, 1)
    return model


def train_main(args, interface):
    model = create_model()
    pipeline = create_pipeline(args.res)

    interface.add_thread(interface.auto_rc)

    with depthai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")

        while True:
            img_rgb = read_latest(q_rgb).getCvFrame()

            # Infer model
            img = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255
            pred = model(img).item()
            pred = math.tanh(pred)
            interface.nn_pred = pred
            print(pred)

            time.sleep(args.infer_time)
