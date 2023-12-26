from typing import Literal, Optional

import yaml

from pydantic import BaseModel
import torch

PRECISION_TYPES = Literal["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"]


class PretrainedModelConfig(BaseModel):
    name_or_path: str
    v2: bool = False
    v_pred: bool = False
    clip_skip: Optional[int] = None


class NetworkConfig(BaseModel):
    rank: int = 1
    alpha: float = 1.0


class TrainConfig(BaseModel):    
    precision: PRECISION_TYPES = "float32"
    noise_scheduler: Literal["ddim", "ddpm", "lms", "euler_a"] = "ddim"

    iterations: int = 3000
    batch_size: int = 1

    lr: float = 1e-4
    unet_lr: float = 1e-4
    text_encoder_lr: float = 5e-5

    optimizer_type: str = "AdamW8bit"
    optimizer_args: list[str] = None

    lr_scheduler: str = "cosine_with_restarts"
    lr_warmup_steps: int = 500
    lr_scheduler_num_cycles: int = 3
    lr_scheduler_power: float = 1.0
    lr_scheduler_args: str = ""

    max_grad_norm: float = 0.0

    max_denoising_steps: int = 30


class SaveConfig(BaseModel):
    name: str = "untitled"
    path: str = "./output"
    per_steps: int = 500
    precision: PRECISION_TYPES = "float32"


class LoggingConfig(BaseModel):
    use_wandb: bool = False
    run_name: str = None
    verbose: bool = False
    
    interval: int = 50
    prompts: list[str] = []
    negative_prompt: str = "bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,low quality,long neck,lowres,error,blurry,missing fingers,fewer digits,missing arms,text,cropped,Humpbacked,bad hands,username"
    anchor_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int = None
    generate_num: int = 1
    eval_num: int = 10
    
class InferenceConfig(BaseModel):
    use_wandb: bool = False
    negative_prompt: str = "bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,low quality,long neck,lowres,error,blurry,missing fingers,fewer digits,missing arms,text,cropped,Humpbacked,bad hands,username"
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seeds: list[int] = None    
    precision: PRECISION_TYPES = "float32"

class OtherConfig(BaseModel):
    use_xformers: bool = False


class RootConfig(BaseModel):
    prompts_file: Optional[str] = None
    
    pretrained_model: PretrainedModelConfig

    network: Optional[NetworkConfig] = None

    train: Optional[TrainConfig] = None

    save: Optional[SaveConfig] = None

    logging: Optional[LoggingConfig] = None

    inference: Optional[InferenceConfig] = None

    other: Optional[OtherConfig] = None


def parse_precision(precision: str) -> torch.dtype:
    if precision == "fp32" or precision == "float32":
        return torch.float32
    elif precision == "fp16" or precision == "float16":
        return torch.float16
    elif precision == "bf16" or precision == "bfloat16":
        return torch.bfloat16

    raise ValueError(f"Invalid precision type: {precision}")


def load_config_from_yaml(config_path: str) -> RootConfig:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    root = RootConfig(**config)

    if root.train is None:
        root.train = TrainConfig()

    if root.save is None:
        root.save = SaveConfig()

    if root.logging is None:
        root.logging = LoggingConfig()

    if root.inference is None:
        root.inference = InferenceConfig()

    if root.other is None:
        root.other = OtherConfig()

    return root
