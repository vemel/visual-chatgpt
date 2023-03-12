import random

import cv2
import einops
import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything

from ControlNet.annotator.util import HWC3, resize_image
from ControlNet.cldm.ddim_hacked import DDIMSampler

from .base import BaseTool


class Seg2Image(BaseTool):
    def __init__(self, device: str) -> None:
        print("Initialize the seg2image model...")
        self.model = self.create_model(
            "ControlNet/models/cldm_v15.yaml",
            device,
            "ControlNet/models/control_sd15_seg.pth",
        )
        self.device = device
        self.ddim_sampler = DDIMSampler(self.model)
        self.ddim_steps = 20
        self.image_resolution = 512
        self.num_samples = 1
        self.save_memory = False
        self.strength = 1.0
        self.guess_mode = False
        self.scale = 9.0
        self.seed = -1
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    def inference(self, inputs: str) -> str:
        print("===>Starting seg2image Inference")
        image_path, instruct_text = inputs.split(",")[0], ",".join(
            inputs.split(",")[1:]
        )
        image = Image.open(image_path)
        image = np.array(image)
        prompt = instruct_text
        img = resize_image(HWC3(image), self.image_resolution)
        H, W, C = img.shape
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        control = torch.from_numpy(img.copy()).float().to(device=self.device) / 255.0
        control = torch.stack([control for _ in range(self.num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        cond = {
            "c_concat": [control],
            "c_crossattn": [
                self.model.get_learned_conditioning(
                    [prompt + ", " + self.a_prompt] * self.num_samples
                )
            ],
        }
        un_cond = {
            "c_concat": None if self.guess_mode else [control],
            "c_crossattn": [
                self.model.get_learned_conditioning([self.n_prompt] * self.num_samples)
            ],
        }
        shape = (4, H // 8, W // 8)
        self.model.control_scales = (
            [self.strength * (0.825 ** float(12 - i)) for i in range(13)]
            if self.guess_mode
            else ([self.strength] * 13)
        )
        samples, intermediates = self.ddim_sampler.sample(
            self.ddim_steps,
            self.num_samples,
            shape,
            cond,
            verbose=False,
            eta=0.0,
            unconditional_guidance_scale=self.scale,
            unconditional_conditioning=un_cond,
        )
        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )
        real_image = Image.fromarray(x_samples[0])  # default the index0 image
        updated_image_path = self.save_image(
            real_image, image_path, func_name="segment2image"
        )
        return updated_image_path
