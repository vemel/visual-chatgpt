from .base import BaseTool
from transformers import (
    BlipForQuestionAnswering,
    BlipProcessor
)
from PIL import Image

class BLIPVQA(BaseTool):
    def __init__(self, device: str) -> None:
        print("Initializing BLIP VQA to %s" % device)
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base"
        ).to(self.device)

    def get_answer_from_question_and_image(self, inputs: str) -> str:
        image_path, question = inputs.split(",")
        raw_image = Image.open(image_path).convert("RGB")
        print(f"BLIPVQA :question :{question}")
        inputs = self.processor(raw_image, question, return_tensors="pt").to(
            self.device
        )
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer