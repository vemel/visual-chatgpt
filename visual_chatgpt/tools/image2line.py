import cv2
import numpy as np
from PIL import Image

from ControlNet.annotator.mlsd import MLSDdetector
from ControlNet.annotator.util import HWC3, resize_image

from .base import BaseTool


class Image2Line(BaseTool):
    def __init__(self) -> None:
        print("Direct detect straight line...")
        self.detector = MLSDdetector()
        self.value_thresh = 0.1
        self.dis_thresh = 0.1
        self.resolution = 512

    def inference(self, inputs: str) -> str:
        print("===>Starting image2hough Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        hough = self.detector(
            resize_image(image, self.resolution), self.value_thresh, self.dis_thresh
        )
        hough = 255 - cv2.dilate(
            hough, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1
        )
        image = Image.fromarray(hough)
        return self.save_image(image, inputs, func_name="line-of")
