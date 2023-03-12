import cv2
import numpy as np
from PIL import Image

from ControlNet.annotator.hed import HEDdetector, nms
from ControlNet.annotator.util import HWC3, resize_image

from .base import BaseTool


class Image2Scribble(BaseTool):
    def __init__(self) -> None:
        print("Direct detect scribble.")
        self.detector = HEDdetector()
        self.resolution = 512

    def inference(self, inputs: str) -> str:
        print("===>Starting image2scribble Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        detected_map = self.detector(resize_image(image, self.resolution))
        detected_map = HWC3(detected_map)
        image = resize_image(image, self.resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        detected_map = nms(detected_map, 127, 3.0)
        detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
        detected_map[detected_map > 4] = 255
        detected_map[detected_map < 255] = 0
        detected_map = 255 - detected_map
        image = Image.fromarray(detected_map)
        updated_image_path = self.save_image(image, inputs, func_name="scribble")
        return updated_image_path
