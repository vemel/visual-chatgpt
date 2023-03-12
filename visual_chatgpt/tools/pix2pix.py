from PIL import Image
import torch
from visual_chatgpt.utils import get_new_image_name
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from .base import BaseTool

class Pix2Pix(BaseTool):
    def __init__(self, device: str) -> None:
        print("Initializing Pix2Pix to %s" % device)
        self.device = device
        model: StableDiffusionInstructPix2PixPipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained( # type: ignore
            "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None
        )
        self.pipe = model.to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

    def inference(self, inputs: str) -> str:
        """Change style of image."""
        print("===>Starting Pix2Pix Inference")
        image_path, instruct_text = inputs.split(",")[0], ",".join(
            inputs.split(",")[1:]
        )
        original_image = Image.open(image_path)
        image = self.pipe(
            instruct_text,
            image=original_image,
            num_inference_steps=40,
            image_guidance_scale=1.2,
        ).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="pix2pix")
        image.save(updated_image_path)
        return updated_image_path
