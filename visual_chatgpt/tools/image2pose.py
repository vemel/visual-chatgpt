from .base import BaseTool
from PIL import Image
import numpy as np
from ControlNet.annotator.util import HWC3, resize_image
import cv2
from ControlNet.annotator.openpose import OpenposeDetector


class Image2Pose(BaseTool):
    def __init__(self) -> None:
        print("Direct human pose.")
        self.detector = OpenposeDetector()
        self.resolution = 512

    def inference(self, inputs: str) -> str:
        print("===>Starting image2pose Inference")
        image = Image.open(inputs)
        image = np.array(image)
        image = HWC3(image)
        detected_map, _ = self.detector(resize_image(image, self.resolution))
        detected_map = HWC3(detected_map)
        image = resize_image(image, self.resolution)
        H, W, C = image.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        updated_image_path = self.get_new_image_name(inputs, func_name="human-pose")
        image = Image.fromarray(detected_map)
        image.save(updated_image_path)
        return updated_image_path