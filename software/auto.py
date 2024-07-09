import time

import torch

from camera import *
from train import AutocarModel, DEVICE, preprocess_data


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
        q_rgb = device.getOutputQueue("rgb")
        q_depth = device.getOutputQueue("depth")
        q_depth_conf = device.getOutputQueue("depth_conf")

        while True:
            img_rgb = read_latest(q_rgb).getCvFrame()
            img_depth = read_latest(q_depth).getFrame()
            img_depth = (img_depth / np.max(img_depth) * 255).astype(np.uint8)
            img_depth_conf = read_latest(q_depth_conf).getFrame()

            nn_enabled = interface.rc_values[5] > 0.5

            # Infer model
            if nn_enabled:
                with torch.no_grad():
                    color = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255
                    depth = torch.tensor(img_depth).unsqueeze(0).float() / 255
                    depth_conf = torch.tensor(img_depth_conf).unsqueeze(0).float() / 255
                    x = preprocess_data(color, depth, depth_conf)
                    pred = model(x).item()
            else:
                pred = 0
            interface.nn_pred = pred

            time.sleep(0.01)
