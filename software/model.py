import math
import random
import time
from socket import socket, AF_INET, SOCK_STREAM
from threading import Thread

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


def worker_infer(args, objs):
    while True:
        if objs["img_rgb"] is not None:
            img = torch.tensor(objs["img_rgb"]).permute(2, 0, 1).unsqueeze(0).float() / 255
            pred = objs["model"](img).item()
            pred = math.tanh(pred)
            objs["interface"].nn_pred = pred
            objs["pred"] = pred

        time.sleep(args.infer_ival)


def worker_new_data(args, objs):
    while True:
        online_enabled = objs["interface"].rc_values[5] > 0.5
        if online_enabled and objs["img_rgb"] is not None:
            if random.random() < abs(objs["interface"].rc_values[0] - 0.5) + 0.5:
                post_new_data(args, objs["img_rgb"], objs["interface"].rc_values[0] * 2 - 1 + objs["pred"])
            else:
                time.sleep(1)


def worker_new_model(args, objs):
    while True:
        online_enabled = objs["interface"].rc_values[5] > 0.5
        if online_enabled:
            conn = create_conn(args)
            sendobj(conn, {"type": "get_model"})
            model_data = recvobj(conn)["model"]
            conn.close()

            if model_data is not None:
                with open("/tmp/model.pt", "wb") as f:
                    f.write(model_data)
                try:
                    objs["model"].load_state_dict(torch.load("/tmp/model.pt", map_location=DEVICE))
                except Exception as e:
                    print("Model update failed:", e)
                else:
                    print("Model updated.")


def train_main(args, interface):
    """
    Main for reinforcement learning client (car).
    """
    model = create_model()
    pipeline = create_pipeline(args.res)

    interface.add_thread(interface.auto_rc)

    objs = {
        "model": model,
        "interface": interface,
        "pred": 0,
        "img_rgb": None,
    }

    with depthai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")

        threads = []
        threads.append(Thread(target=worker_infer, args=(args, objs)))
        threads.append(Thread(target=worker_new_data, args=(args, objs)))
        threads.append(Thread(target=worker_new_model, args=(args, objs)))
        for thread in threads:
            thread.start()

        while True:
            img_rgb = read_latest(q_rgb).getCvFrame()
            objs["img_rgb"] = img_rgb
