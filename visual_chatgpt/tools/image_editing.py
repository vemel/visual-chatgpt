from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from .base import BaseTool
from .mask_former import MaskFormer


class ImageEditing(BaseTool):
    def __init__(self, device: str) -> None:
        print("Initializing StableDiffusionInpaint to %s" % device)
        self.device = device
        self.mask_former = MaskFormer(device=self.device)

        model: StableDiffusionInpaintPipeline = StableDiffusionInpaintPipeline.from_pretrained(  # type: ignore
            "runwayml/stable-diffusion-inpainting",
        )
        self.inpainting = model.to(device)

    def remove_part_of_image(self, input):
        image_path, to_be_removed_txt = input.split(",")
        print(f"remove_part_of_image: to_be_removed {to_be_removed_txt}")
        return self.replace_part_of_image(
            f"{image_path},{to_be_removed_txt},background"
        )

    def replace_part_of_image(self, input):
        image_path, to_be_replaced_txt, replace_with_txt = input.split(",")
        print(f"replace_part_of_image: replace_with_txt {replace_with_txt}")
        original_image = Image.open(image_path)
        mask_image = self.mask_former.inference(image_path, to_be_replaced_txt)
        updated_image = self.inpainting(
            prompt=replace_with_txt, image=original_image, mask_image=mask_image
        ).images[0]
        updated_image_path = self.save_image(
            updated_image, image_path, func_name="replace-something"
        )
        return updated_image_path
