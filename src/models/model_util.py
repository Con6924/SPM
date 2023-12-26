from typing import Literal, Union, Optional

import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers import (
    UNet2DConditionModel,
    SchedulerMixin,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    AltDiffusionPipeline,
    DiffusionPipeline,
)
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)

TOKENIZER_V1_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
TOKENIZER_V2_MODEL_NAME = "stabilityai/stable-diffusion-2-1"

AVAILABLE_SCHEDULERS = Literal["ddim", "ddpm", "lms", "euler_a"]

SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]

DIFFUSERS_CACHE_DIR = ".cache/"  # if you want to change the cache dir, change this
LOCAL_ONLY = False  # if you want to use only local files, change this


def load_diffusers_model(
    pretrained_model_name_or_path: str,
    v2: bool = False,
    clip_skip: Optional[int] = None,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel,]:

    if v2:
        tokenizer = CLIPTokenizer.from_pretrained(
            TOKENIZER_V2_MODEL_NAME,
            subfolder="tokenizer",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            # default is clip skip 2
            num_hidden_layers=24 - (clip_skip - 1) if clip_skip is not None else 23,
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        )
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            TOKENIZER_V1_MODEL_NAME,
            subfolder="tokenizer",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            num_hidden_layers=12 - (clip_skip - 1) if clip_skip is not None else 12,
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        )

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )

    return tokenizer, text_encoder, unet


def load_checkpoint_model(
    checkpoint_path: str,
    v2: bool = False,
    clip_skip: Optional[int] = None,
    weight_dtype: torch.dtype = torch.float32,
    device = "cuda",
) -> tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel, DiffusionPipeline]:
    print(f"Loading checkpoint from {checkpoint_path}")
    if checkpoint_path == "BAAI/AltDiffusion":
        pipe = AltDiffusionPipeline.from_pretrained(
            "BAAI/AltDiffusion", 
            upcast_attention=True if v2 else False,
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
            local_files_only=LOCAL_ONLY,
        ).to(device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            checkpoint_path,
            upcast_attention=True if v2 else False,
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
            local_files_only=LOCAL_ONLY,
        ).to(device)

    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    if clip_skip is not None:
        if v2:
            text_encoder.config.num_hidden_layers = 24 - (clip_skip - 1)
        else:
            text_encoder.config.num_hidden_layers = 12 - (clip_skip - 1)

    return tokenizer, text_encoder, unet, pipe


def load_models(
    pretrained_model_name_or_path: str,
    scheduler_name: AVAILABLE_SCHEDULERS,
    v2: bool = False,
    v_pred: bool = False,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel, SchedulerMixin, DiffusionPipeline, ]:
    tokenizer, text_encoder, unet, pipe = load_checkpoint_model(
        pretrained_model_name_or_path, v2=v2, weight_dtype=weight_dtype
    )

    scheduler = create_noise_scheduler(
        scheduler_name,
        prediction_type="v_prediction" if v_pred else "epsilon",
    )

    return tokenizer, text_encoder, unet, scheduler, pipe


def load_diffusers_model_xl(
    pretrained_model_name_or_path: str,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[list[CLIPTokenizer], list[SDXL_TEXT_ENCODER_TYPE], UNet2DConditionModel,]:
    # returns tokenizer, tokenizer_2, text_encoder, text_encoder_2, unet

    tokenizers = [
        CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        ),
        CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
            pad_token_id=0,  # same as open clip
        ),
    ]

    text_encoders = [
        CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        ),
        CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
        ),
    ]

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
    )

    return tokenizers, text_encoders, unet, None


def load_checkpoint_model_xl(
    checkpoint_path: str,
    weight_dtype: torch.dtype = torch.float32,
    device = "cuda",
) -> tuple[list[CLIPTokenizer], list[SDXL_TEXT_ENCODER_TYPE], UNet2DConditionModel, DiffusionPipeline, ]:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        checkpoint_path,
        torch_dtype=weight_dtype,
        cache_dir=DIFFUSERS_CACHE_DIR,
        local_files_only=LOCAL_ONLY,
    ).to(device)

    unet = pipe.unet
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    if len(text_encoders) == 2:
        text_encoders[1].pad_token_id = 0

    return tokenizers, text_encoders, unet, pipe


def load_models_xl(
    pretrained_model_name_or_path: str,
    scheduler_name: AVAILABLE_SCHEDULERS,
    weight_dtype: torch.dtype = torch.float32,
) -> tuple[
    list[CLIPTokenizer],
    list[SDXL_TEXT_ENCODER_TYPE],
    UNet2DConditionModel,
    SchedulerMixin,
    DiffusionPipeline,
]:
    (
        tokenizers,
        text_encoders,
        unet,
        pipe,
    ) = load_checkpoint_model_xl(pretrained_model_name_or_path, weight_dtype)

    scheduler = create_noise_scheduler(scheduler_name)

    return tokenizers, text_encoders, unet, scheduler, pipe


def create_noise_scheduler(
    scheduler_name: AVAILABLE_SCHEDULERS = "ddpm",
    prediction_type: Literal["epsilon", "v_prediction"] = "epsilon",
) -> SchedulerMixin:
    
    name = scheduler_name.lower().replace(" ", "_")
    if name == "ddim":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddim
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type=prediction_type,  # これでいいの？
        )
    elif name == "ddpm":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddpm
        scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type=prediction_type,
        )
    elif name == "lms":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/lms_discrete
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=prediction_type,
        )
    elif name == "euler_a":
        # https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/euler_ancestral
        scheduler = EulerAncestralDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=prediction_type,
        )
    else:
        raise ValueError(f"Unknown scheduler name: {name}")

    return scheduler
