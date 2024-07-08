import math
import random
import time
from socket import socket, AF_INET, SOCK_STREAM

import cv2

import torch
import torchvision

from camera import *
from conn import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    last_new_model = 0
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

            online_enabled = interface.rc_values[5] > 0.5

            # Check for new data.
            if online_enabled:# and time.time() - last_new_data > args.new_data_ival:
                last_new_data = time.time()
                if random.random() < abs(interface.rc_values[0] - 0.5) + 0.5:
                    post_new_data(args, img_rgb, interface.rc_values[0] * 2 - 1 + pred)

            # Check for new model.
            if online_enabled and time.time() - last_new_model > args.new_model_ival:
                last_new_model = time.time()

                conn = create_conn(args)
                sendobj(conn, {"type": "get_model"})
                model_data = recvobj(conn)["model"]
                conn.close()

                if model_data is not None:
                    with open("/tmp/model.pt", "wb") as f:
                        f.write(model_data)
                    try:
                        model.load_state_dict(torch.load("/tmp/model.pt", map_location=DEVICE))
                    except Exception as e:
                        print("Model update failed:", e)
                    else:
                        print("Model updated.")

            time.sleep(args.infer_ival)
