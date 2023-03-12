from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from .base import BaseTool

class ImageCaptioning(BaseTool):
    def __init__(self, device: str) -> None:
        print("Initializing ImageCaptioning to %s" % device)
        self.device = device
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        model: BlipForConditionalGeneration = BlipForConditionalGeneration.from_pretrained(  # type: ignore
            "Salesforce/blip-image-captioning-base"
        )
        self.model = model.to(self.device)

    def inference(self, image_path: str) -> str:
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(
            self.device
        )
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        return captions