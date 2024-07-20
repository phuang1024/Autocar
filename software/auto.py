import time

import depthai
import numpy as np
import torch

from camera import *
from train import AutocarModel, DEVICE


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

    interface.add_thread(interface.auto_rc)

    with depthai.Device(pipeline) as device:
        wrapper = PipelineWrapper(device, include_nn=is_onnx)

        em = torch.randn(1, model.em_size).to(DEVICE)

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
                        pred, curr_em = model(x, em)
                        pred = pred.item()
                        em = 0.7 * em + curr_em

                print("Pred", pred)

            else:
                pred = 0

            interface.nn_pred = pred

            time.sleep(0.01)
