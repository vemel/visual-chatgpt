import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .base import BaseTool


class T2I(BaseTool):
    def __init__(self, device: str) -> None:
        print("Initializing T2I to %s" % device)
        self.device = device
        self.pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(  # type: ignore
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        self.text_refine_tokenizer = AutoTokenizer.from_pretrained(
            "Gustavosta/MagicPrompt-Stable-Diffusion"
        )
        self.text_refine_model = AutoModelForCausalLM.from_pretrained(
            "Gustavosta/MagicPrompt-Stable-Diffusion"
        )
        self.text_refine_gpt2_pipe = pipeline(
            "text-generation",
            model=self.text_refine_model,
            tokenizer=self.text_refine_tokenizer,
            device=self.device,
        )
        self.pipe.to(device)

    def inference(self, text: str) -> str:
        image_filename = self.get_image_filename()
        refined_text = self.text_refine_gpt2_pipe(text)[0]["generated_text"]
        print(f"{text} refined to {refined_text}")
        image = self.pipe(refined_text).images[0]
        image.save(image_filename)
        print(f"Processed T2I.run, text: {text}, image_filename: {image_filename}")
        return image_filename
