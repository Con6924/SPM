from pydantic import BaseModel
import torch
import yaml

class GenerationConfig(BaseModel):
    prompts: list[str] = []
    negative_prompt: str = "bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,low quality,long neck,lowres,error,blurry,missing fingers,fewer digits,missing arms,text,cropped,Humpbacked,bad hands,username"
    unconditional_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int = 2024
    generate_num: int = 1

    save_path: str = None  # can be a template, e.g. "path/to/img_{}.png",
    # then the generated images will be saved as "path/to/img_0.png", "path/to/img_1.png", ...

    def dict(self):
        results = {}
        for attr in vars(self):
            if not attr.startswith("_"):
                results[attr] = getattr(self, attr)
        return results
    
    @staticmethod
    def fix_format(cfg):
        for k, v in cfg.items():
            if isinstance(v, list):
                cfg[k] = v[0]
            elif isinstance(v, torch.Tensor):
                cfg[k] = v.item()

def load_config_from_yaml(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return GenerationConfig(**cfg)
