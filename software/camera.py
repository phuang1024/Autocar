import time

import cv2
import depthai


def create_pipeline(res):
    pipeline = depthai.Pipeline()

    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(res, res)
    cam_rgb.setInterleaved(False)

    depth_left = pipeline.createMonoCamera()
    depth_left.setCamera("left")
    depth_left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)

    depth_right = pipeline.createMonoCamera()
    depth_right.setCamera("right")
    depth_right.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)

    depth = pipeline.createStereoDepth()
    depth.setLeftRightCheck(True)
    depth.setExtendedDisparity(False)
    depth.setSubpixel(False)
    depth_left.out.link(depth.left)
    depth_right.out.link(depth.right)

    """
    imu = pipeline.createIMU()
    imu.enableIMUSensor([depthai.IMUSensor.ROTATION_VECTOR], 10)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(1)
    """

    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    xout_depth = pipeline.createXLinkOut()
    xout_depth.setStreamName("depth")
    depth.disparity.link(xout_depth.input)

    xout_depth_conf = pipeline.createXLinkOut()
    xout_depth_conf.setStreamName("depth_conf")
    depth.confidenceMap.link(xout_depth_conf.input)

    """
    xout_imu = pipeline.createXLinkOut()
    xout_imu.setStreamName("imu")
    imu.out.link(xout_imu.input)
    """

    return pipeline


class PipelineWrapper:
    """
    Handles creating and reading all queues.
    """

    def __init__(self, device):
        self.device = device
        self.queues = {}
        for name in ["rgb", "depth", "depth_conf"]:
            self.queues[name] = self.device.getOutputQueue(name)

    def get(self):
        return {
            "rgb": read_latest(self.queues["rgb"]).getCvFrame(),
            "depth": crop_resize(read_latest(self.queues["depth"]).getFrame()),
            "depth_conf": crop_resize(read_latest(self.queues["depth_conf"]).getFrame()),
        }


def crop_resize(img):
    diff = img.shape[1] - img.shape[0]
    img = img[diff // 2 : -diff // 2]
    img = cv2.resize(img, (256, 256))
    return img


def read_latest(queue):
    data = None
    while queue.has() or data is None:
        data = queue.get()
        time.sleep(0.01)
    return data
