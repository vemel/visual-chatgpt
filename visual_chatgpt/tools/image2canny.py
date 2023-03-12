import numpy as np
from PIL import Image

from ControlNet.annotator.canny import CannyDetector

from .base import BaseTool


class Image2Canny(BaseTool):
    def __init__(self) -> None:
        print("Direct detect canny.")
        self.detector = CannyDetector()
        self.low_thresh = 100
        self.high_thresh = 200

    def inference(self, inputs: str) -> str:
        print("===>Starting image2canny Inference")
        image = Image.open(inputs)
        image = np.array(image)
        canny = self.detector(image, self.low_thresh, self.high_thresh)
        canny = 255 - canny
        updated_image_path = self.save_image(
            Image.fromarray(canny), inputs, func_name="edge"
        )
        return updated_image_path
