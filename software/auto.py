import time

import torch

from camera import *
from train import AutocarModel, DEVICE


def auto_main(args, interface):
    """
    Main for auto driving.
    """
    model = AutocarModel().to(DEVICE)
    model.load_state_dict(torch.load(args.model_path))
    print("Load from model:", args.model_path)
    pipeline = create_pipeline(args.res)

    interface.add_thread(interface.auto_rc)

    with depthai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")

        while True:
            img_rgb = read_latest(q_rgb).getCvFrame()

            nn_enabled = interface.rc_values[5] > 0.5

            # Infer model
            if nn_enabled:
                with torch.no_grad():
                    img = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255
                    pred = model(img).item()
            else:
                pred = 0
            interface.nn_pred = pred

            time.sleep(0.01)
