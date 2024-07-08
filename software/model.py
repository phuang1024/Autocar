import math
import random
import time
from socket import socket, AF_INET, SOCK_STREAM

import cv2

import torch
import torchvision

from camera import *
from conn import *


def create_model():
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(512, 1)
    return model


def create_conn(args):
    conn = socket(AF_INET, SOCK_STREAM)
    conn.connect((args.ip, args.port))
    return conn


def post_new_data(args, img, label):
    conn = create_conn(args)
    cv2.imwrite("/tmp/img.jpg", img)
    with open("/tmp/img.jpg", "rb") as f:
        img_data = f.read()
    sendobj(conn, {"type": "new_data", "img": img_data, "label": label})


def train_main(args, interface):
    """
    Main for reinforcement learning client (car).
    """
    model = create_model()
    pipeline = create_pipeline(args.res)

    interface.add_thread(interface.auto_rc)

    last_new_data = 0
    with depthai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")

        while True:
            img_rgb = read_latest(q_rgb).getCvFrame()

            # Infer model
            img = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255
            pred = model(img).item()
            pred = math.tanh(pred)
            interface.nn_pred = pred
            #print("pred", pred)

            # Check for new data.
            if time.time() - last_new_data > args.new_data_ival:
                last_new_data = time.time()
                if random.random() < abs(interface.rc_values[0] - 0.5) + 0.2:
                    post_new_data(args, img_rgb, interface.rc_values[0] * 2 - 1)

            time.sleep(args.infer_ival)
