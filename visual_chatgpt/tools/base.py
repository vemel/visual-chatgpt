from pathlib import Path
import uuid
from visual_chatgpt.utils import get_new_image_name
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from ControlNet.cldm.model import load_state_dict
from typing import Any


class BaseTool:
    IMAGE_PATH = Path.cwd() / "image"

    def get_image_filename(self) -> str:
        filename = str(uuid.uuid4())[0:8] + ".png"
        path = self.IMAGE_PATH / filename
        return str(path)
    
    def get_new_image_name(self, org_img_name: str, func_name: str="update") -> str:
        return get_new_image_name(org_img_name, func_name)

    def create_model(self, config_path: str, device: str, state_path: str) -> Any:
        config = OmegaConf.load(config_path)
        OmegaConf.update(config, "model.params.cond_stage_config.params.device", device)
        model = instantiate_from_config(config.model).cpu()
        print(f"Loaded model config from [{config_path}]")
        model = model.to(device)

        model.load_state_dict(
            load_state_dict(state_path, location="cpu")
        )

        return model.to(device)