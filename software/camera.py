import time

import depthai


def create_pipeline(res):
    pipeline = depthai.Pipeline()

    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(res, res)
    cam_rgb.setInterleaved(False)

    """
    imu = pipeline.createIMU()
    imu.enableIMUSensor([depthai.IMUSensor.ROTATION_VECTOR], 10)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(1)
    """

    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    xout_imu = pipeline.createXLinkOut()
    xout_imu.setStreamName("imu")
    #imu.out.link(xout_imu.input)

    return pipeline


def read_latest(queue):
    data = None
    while queue.has() or data is None:
        data = queue.get()
        time.sleep(0.01)
    return data

