import uuid
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from PIL import Image

from ControlNet.cldm.model import load_state_dict
from ControlNet.ldm.util import instantiate_from_config
from visual_chatgpt.utils import get_new_image_name


class BaseTool:
    IMAGE_PATH = Path.cwd() / "image"

    @classmethod
    def get_image_filename(cls) -> str:
        filename = str(uuid.uuid4())[0:8] + ".png"
        path = cls.IMAGE_PATH / filename
        return str(path)

    def get_new_image_name(self, org_img_name: str, func_name: str = "update") -> str:
        return get_new_image_name(org_img_name, func_name)

    def save_image(
        self, image: Image.Image, orig_img_name: str, func_name: str = "update"
    ) -> str:
        if not self.IMAGE_PATH.exists():
            self.IMAGE_PATH.mkdir(parents=True)

        updated_image_path = self.get_new_image_name(orig_img_name, func_name=func_name)
        image.save(updated_image_path)
        return updated_image_path

    def create_model(self, config_path: str, device: str, state_path: str) -> Any:
        config = OmegaConf.load(config_path)
        OmegaConf.update(config, "model.params.cond_stage_config.params.device", device)
        model = instantiate_from_config(config.model).cpu()
        print(f"Loaded model config from [{config_path}]")
        model = model.to(device)

        model.load_state_dict(load_state_dict(state_path, location="cpu"))

        return model.to(device)
