import time

import torch

from camera import *
from train import AutocarModel, DEVICE


def auto_main(args, interface):
    """
    Main for auto driving.
    """
    model = AutocarModel().to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    print("Load from model:", args.model_path)
    pipeline = create_pipeline(args.res)

    interface.add_thread(interface.auto_rc)

    with depthai.Device(pipeline) as device:
        wrapper = PipelineWrapper(device)

        while True:
            images = wrapper.get()

            nn_enabled = interface.rc_values[5] > 0.5

            # Infer model
            if nn_enabled:
                with torch.no_grad():
                    color = torch.tensor(images["rgb"]).permute(2, 0, 1).float() / 255
                    color = torch.mean(color, dim=0)
                    depth = torch.tensor(images["depth"]).float() / 255
                    depth_conf = torch.tensor(images["depth_conf"]).float() / 255
                    x = torch.stack([color, depth, depth_conf], dim=0).unsqueeze(0).to(DEVICE)
                    pred = model(x).item()
            else:
                pred = 0
            interface.nn_pred = pred

            time.sleep(0.01)
