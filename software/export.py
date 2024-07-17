"""
Export torch model to ONNX.
"""

import argparse

import torch

from train import OnnxAutocarModel, DEVICE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Path to the model file.")
    parser.add_argument("output", type=str, help="Path to the output ONNX file.")
    args = parser.parse_args()

    model = OnnxAutocarModel()
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model.eval()

    rgb_shape = torch.empty(1, 3, 256, 256)
    depth_shape = torch.empty(1, 1, 256, 256)
    torch.onnx.export(
        model,
        (rgb_shape, depth_shape),
        args.output,
        input_names=["rgb", "depth"],
        verbose=True
    )


if __name__ == "__main__":
    main()
