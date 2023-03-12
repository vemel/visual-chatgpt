from .base import BaseTool
from PIL import Image
import numpy as np
from ControlNet.annotator.canny import CannyDetector



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
        image = Image.fromarray(canny)
        updated_image_path = self.get_new_image_name(inputs, func_name="edge")
        image.save(updated_image_path)
        return updated_image_path