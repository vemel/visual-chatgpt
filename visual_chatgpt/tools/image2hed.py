import numpy as np
from PIL import Image

from ControlNet.annotator.hed import HEDdetector
from ControlNet.annotator.util import HWC3, resize_image

from .base import BaseTool


class Image2Hed(BaseTool):
    def __init__(self) -> None:
        print("Direct detect soft HED boundary...")
        self.detector = HEDdetector()
        self.resolution = 512

    def inference(self, inputs: str) -> str:
        print("===>Starting image2hed Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        hed = self.detector(resize_image(image, self.resolution))
        image = Image.fromarray(hed)
        return self.save_image(image, inputs, func_name="hed-boundary")
