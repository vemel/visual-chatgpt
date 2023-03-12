from .base import BaseTool
from PIL import Image
import numpy as np
from ControlNet.annotator.util import HWC3, resize_image
import cv2
from ControlNet.annotator.mlsd import MLSDdetector

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
        updated_image_path = self.get_new_image_name(inputs, func_name="line-of")
        hough = 255 - cv2.dilate(
            hough, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1
        )
        image = Image.fromarray(hough)
        image.save(updated_image_path)
        return updated_image_path