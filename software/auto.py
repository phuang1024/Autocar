import time

import torch

from camera import *
from train import AutocarModel, DEVICE, RNN_LENGTH, RNN_DECAY

EVAL_RNN_STRIDE = 4


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

        xs = [None] * ((RNN_LENGTH - 1) * EVAL_RNN_STRIDE + 1)

        while True:
            images = wrapper.get()

            nn_enabled = interface.rc_values[5] > 0.5

            # Infer model
            if nn_enabled:
                with torch.no_grad():
                    # Read new image
                    x = images_to_tensor(images)
                    x = x.float() / 255
                    xs.insert(0, x)
                    xs.pop()

                    # Prepare RNN stack.
                    strided_x = []
                    decay = 1
                    for i in range(RNN_LENGTH):
                        x = xs[i * EVAL_RNN_STRIDE]
                        if x is None:
                            x = torch.zeros_like(xs[0])
                        strided_x.append(x * decay)
                        decay *= RNN_DECAY
                    x = torch.cat(strided_x, dim=0).unsqueeze(0).to(DEVICE)

                    # Predict
                    pred = model(x).item()
                    print("Pred", pred)

            else:
                pred = 0

            interface.nn_pred = pred

            time.sleep(0.01)
