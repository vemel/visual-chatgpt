from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from .base import BaseTool


class MaskFormer(BaseTool):
    def __init__(self, device: str) -> None:
        self.device = device
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        model: CLIPSegForImageSegmentation = CLIPSegForImageSegmentation.from_pretrained(  # type: ignore
            "CIDAS/clipseg-rd64-refined"
        )
        self.model = model.to(device)

    def inference(self, image_path: str, text: str) -> Optional[Image.Image]:
        threshold = 0.5
        min_area = 0.02
        padding = 20
        original_image = Image.open(image_path)
        image = original_image.resize((512, 512))
        inputs = self.processor(
            text=text,
            images=image,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        mask = torch.sigmoid(outputs[0]).squeeze().cpu().numpy() > threshold
        area_ratio = len(np.argwhere(mask)) / (mask.shape[0] * mask.shape[1])
        if area_ratio < min_area:
            return None
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(
                slice(max(0, i - padding), i + padding + 1) for i in idx
            )
            mask_array[padded_slice] = True
        visual_mask = (mask_array * 255).astype(np.uint8)
        image_mask = Image.fromarray(visual_mask)
        return image_mask.resize(image.size)
