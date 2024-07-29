import cv2
import depthai
import torch

FPS = 20


def create_pipeline(res, nn_path=None):
    """
    nn_path: None means no nn.
    """
    pipeline = depthai.Pipeline()

    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(res, res)
    cam_rgb.setFps(FPS)
    cam_rgb.setInterleaved(False)

    st_left = pipeline.createMonoCamera()
    st_left.setFps(FPS)
    st_left.setCamera("left")
    st_left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)

    st_right = pipeline.createMonoCamera()
    st_right.setFps(FPS)
    st_right.setCamera("right")
    st_right.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)

    depth = pipeline.createStereoDepth()
    depth.setLeftRightCheck(True)
    depth.setExtendedDisparity(False)
    depth.setSubpixel(False)
    st_left.out.link(depth.left)
    st_right.out.link(depth.right)

    if nn_path is not None:
        nn = pipeline.createNeuralNetwork()
        nn.setBlobPath(nn_path)
        cam_rgb.preview.link(nn.inputs["rgb"])
        depth.disparity.link(nn.inputs["depth"])

    imu = pipeline.createIMU()
    imu.enableIMUSensor([depthai.IMUSensor.ROTATION_VECTOR], FPS)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(1)

    if nn_path is None:
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        xout_depth_fac = pipeline.createXLinkOut()
        xout_depth_fac.setStreamName("depth_fac")
        depth.disparity.link(xout_depth_fac.input)

        xout_depth_dist = pipeline.createXLinkOut()
        xout_depth_dist.setStreamName("depth_dist")
        depth.depth.link(xout_depth_dist.input)

        xout_depth_conf = pipeline.createXLinkOut()
        xout_depth_conf.setStreamName("depth_conf")
        depth.confidenceMap.link(xout_depth_conf.input)

    else:
        xout_nn = pipeline.createXLinkOut()
        xout_nn.setStreamName("nn")
        nn.out.link(xout_nn.input)

    xout_imu = pipeline.createXLinkOut()
    xout_imu.setStreamName("imu")
    imu.out.link(xout_imu.input)

    return pipeline


class PipelineWrapper:
    """
    Handles creating and reading all queues.
    """

    def __init__(self, device, include_nn=False):
        self.device = device

        if include_nn:
            self.names = ["nn"]
        else:
            self.names = ["rgb", "depth_fac", "depth_dist", "depth_conf"]

        self.queues = {}
        for name in self.names:
            queue = self.device.getOutputQueue(name)
            queue.setMaxSize(1)
            queue.setBlocking(False)
            self.queues[name] = queue

    def get(self):
        ret = {}
        for name in self.names:
            frame = self.queues[name].get()
            if name == "nn":
                frame = frame
            elif name == "rgb":
                frame = frame.getCvFrame()
            else:
                frame = crop_resize(frame.getFrame())
            ret[name] = frame

        return ret


def crop_resize(img):
    """
    img: HWC
    """
    diff = img.shape[1] - img.shape[0]
    img = img[:, diff // 2 : -diff // 2]
    img = cv2.resize(img, (256, 256))
    return img


def images_to_tensor(images):
    """
    Process return of PipelineWrapper.get() into tensor input for model.

    Return:
        (4, 256, 256), CHW
        uint8, 0 to 255
    """
    x = torch.empty((4, 256, 256), dtype=torch.uint8)
    x[0:3] = torch.tensor(images["rgb"]).permute(2, 0, 1)
    x[3] = torch.tensor(images["depth_fac"])
    return x
