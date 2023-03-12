import cv2
import numpy as np
from PIL import Image

from ControlNet.annotator.midas import MidasDetector
from ControlNet.annotator.util import HWC3, resize_image

from .base import BaseTool


class Image2Depth(BaseTool):
    def __init__(self) -> None:
        print("Direct depth estimation.")
        self.detector = MidasDetector()
        self.resolution = 512

    def inference(self, inputs: str) -> str:
        print("===>Starting image2depth Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        detected_map, _ = self.detector(resize_image(image, self.resolution))
        detected_map = HWC3(detected_map)
        image = resize_image(image, self.resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(detected_map)
        updated_image_path = self.save_image(image, inputs, func_name="depth")
        return updated_image_path
