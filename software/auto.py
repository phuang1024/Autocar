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
                    x = images_to_tensor(images)
                    x = x.float() / 255
                    x = x.unsqueeze(0).to(DEVICE)
                    pred = model(x).item()
                    print("Pred", pred)
            else:
                pred = 0
            interface.nn_pred = pred

            time.sleep(0.01)
