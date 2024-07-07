"""
Scripts for generating train data.
"""

import cv2
import depthai


def create_pipeline():
    pipeline = depthai.Pipeline()
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(900, 900)
    cam_rgb.setInterleaved(False)

    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    return pipeline


def gen_data(args, interface):
    pipeline = create_pipeline()
    with depthai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")

        while True:
            while True:
                img_rgb = q_rgb.get()
                if img_rgb is not None:
                    break

            img_rgb = img_rgb.getCvFrame()
            cv2.imshow("rgb", img_rgb)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
