from .base import BaseTool
from PIL import Image
import numpy as np
from ControlNet.annotator.hed import HEDdetector
from ControlNet.annotator.util import HWC3, resize_image


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
        updated_image_path = self.get_new_image_name(inputs, func_name="hed-boundary")
        image = Image.fromarray(hed)
        image.save(updated_image_path)
        return updated_image_path