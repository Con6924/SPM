from typing import Optional, Union

import ast
import importlib
import torch
from torch.optim import Optimizer
import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, SchedulerMixin, DiffusionPipeline
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION

from src.models.model_util import SDXL_TEXT_ENCODER_TYPE

from tqdm import tqdm

UNET_IN_CHANNELS = 4  # Stable Diffusion の in_channels は 4 で固定。XLも同じ。
VAE_SCALE_FACTOR = 8  # 2 ** (len(vae.config.block_out_channels) - 1) = 8

UNET_ATTENTION_TIME_EMBED_DIM = 256  # XL
TEXT_ENCODER_2_PROJECTION_DIM = 1280
UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM = 2816


def get_random_noise(
    batch_size: int, height: int, width: int, generator: torch.Generator = None
) -> torch.Tensor:
    return torch.randn(
        (
            batch_size,
            UNET_IN_CHANNELS,
            height // VAE_SCALE_FACTOR,  # 縦と横これであってるのかわからないけど、どっちにしろ大きな問題は発生しないのでこれでいいや
            width // VAE_SCALE_FACTOR,
        ),
        generator=generator,
        device="cpu",
    )


# https://www.crosslabs.org/blog/diffusion-with-offset-noise
def apply_noise_offset(latents: torch.FloatTensor, noise_offset: float):
    latents = latents + noise_offset * torch.randn(
        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
    )
    return latents


def get_initial_latents(
    scheduler: SchedulerMixin,
    n_imgs: int,
    height: int,
    width: int,
    n_prompts: int,
    generator=None,
) -> torch.Tensor:
    noise = get_random_noise(n_imgs, height, width, generator=generator).repeat(
        n_prompts, 1, 1, 1
    )

    latents = noise * scheduler.init_noise_sigma

    return latents


def text_tokenize(
    tokenizer: CLIPTokenizer,  # 普通ならひとつ、XLならふたつ！
    prompts: list[str],
):
    return tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids


def text_encode(text_encoder: CLIPTextModel, tokens):
    return text_encoder(tokens.to(text_encoder.device))[0]


def encode_prompts(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTokenizer,
    prompts: list[str],
    return_tokens: bool = False,
):
    text_tokens = text_tokenize(tokenizer, prompts)
    text_embeddings = text_encode(text_encoder, text_tokens)

    if return_tokens:
        return text_embeddings, torch.unique(text_tokens, dim=1)
    return text_embeddings


# https://github.com/huggingface/diffusers/blob/78922ed7c7e66c20aa95159c7b7a6057ba7d590d/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L334-L348
def text_encode_xl(
    text_encoder: SDXL_TEXT_ENCODER_TYPE,
    tokens: torch.FloatTensor,
    num_images_per_prompt: int = 1,
):
    prompt_embeds = text_encoder(
        tokens.to(text_encoder.device), output_hidden_states=True
    )
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]  # always penultimate layer

    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompts_xl(
    tokenizers: list[CLIPTokenizer],
    text_encoders: list[SDXL_TEXT_ENCODER_TYPE],
    prompts: list[str],
    num_images_per_prompt: int = 1,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    # text_encoder and text_encoder_2's penuultimate layer's output
    text_embeds_list = []
    pooled_text_embeds = None  # always text_encoder_2's pool

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_tokens_input_ids = text_tokenize(tokenizer, prompts)
        text_embeds, pooled_text_embeds = text_encode_xl(
            text_encoder, text_tokens_input_ids, num_images_per_prompt
        )

        text_embeds_list.append(text_embeds)

    bs_embed = pooled_text_embeds.shape[0]
    pooled_text_embeds = pooled_text_embeds.repeat(1, num_images_per_prompt).view(
        bs_embed * num_images_per_prompt, -1
    )

    return torch.concat(text_embeds_list, dim=-1), pooled_text_embeds


def concat_embeddings(
    unconditional: torch.FloatTensor,
    conditional: torch.FloatTensor,
    n_imgs: int,
):
    return torch.cat([unconditional, conditional]).repeat_interleave(n_imgs, dim=0)


# ref: https://github.com/huggingface/diffusers/blob/0bab447670f47c28df60fbd2f6a0f833f75a16f5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L721
def predict_noise(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,  # 現在のタイムステップ
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
    guidance_scale=7.5,
) -> torch.FloatTensor:
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
    ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    return guided_target


# ref: https://github.com/huggingface/diffusers/blob/0bab447670f47c28df60fbd2f6a0f833f75a16f5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L746
@torch.no_grad()
def diffusion(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,  # ただのノイズだけのlatents
    text_embeddings: torch.FloatTensor,
    total_timesteps: int = 1000,
    start_timesteps=0,
    **kwargs,
):
    # latents_steps = []

    for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps]):
        noise_pred = predict_noise(
            unet, scheduler, timestep, latents, text_embeddings, **kwargs
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    # return latents_steps
    return latents


def rescale_noise_cfg(
    noise_cfg: torch.FloatTensor, noise_pred_text, guidance_rescale=0.0
):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )

    return noise_cfg


def predict_noise_xl(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,  # 現在のタイムステップ
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
    add_text_embeddings: torch.FloatTensor,  # pooled なやつ
    add_time_ids: torch.FloatTensor,
    guidance_scale=7.5,
    guidance_rescale=0.7,
) -> torch.FloatTensor:
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    added_cond_kwargs = {
        "text_embeds": add_text_embeddings,
        "time_ids": add_time_ids,
    }

    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    # https://github.com/huggingface/diffusers/blob/7a91ea6c2b53f94da930a61ed571364022b21044/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L775
    noise_pred = rescale_noise_cfg(
        noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
    )

    return noise_pred


@torch.no_grad()
def diffusion_xl(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,  # ただのノイズだけのlatents
    text_embeddings: tuple[torch.FloatTensor, torch.FloatTensor],
    add_text_embeddings: torch.FloatTensor,  # pooled なやつ
    add_time_ids: torch.FloatTensor,
    guidance_scale: float = 1.0,
    total_timesteps: int = 1000,
    start_timesteps=0,
):
    # latents_steps = []

    for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps]):
        noise_pred = predict_noise_xl(
            unet,
            scheduler,
            timestep,
            latents,
            text_embeddings,
            add_text_embeddings,
            add_time_ids,
            guidance_scale=guidance_scale,
            guidance_rescale=0.7,
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    # return latents_steps
    return latents


# for XL
def get_add_time_ids(
    height: int,
    width: int,
    dynamic_crops: bool = False,
    dtype: torch.dtype = torch.float32,
):
    if dynamic_crops:
        # random float scale between 1 and 3
        random_scale = torch.rand(1).item() * 2 + 1
        original_size = (int(height * random_scale), int(width * random_scale))
        # random position
        crops_coords_top_left = (
            torch.randint(0, original_size[0] - height + 1, (1,)).item(),
            torch.randint(0, original_size[1] - width + 1, (1,)).item(),
        )
        target_size = (height, width)
    else:
        original_size = (height, width)
        crops_coords_top_left = (0, 0)
        target_size = (height, width)

    # this is expected as 6
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    # this is expected as 2816
    passed_add_embed_dim = (
        UNET_ATTENTION_TIME_EMBED_DIM * len(add_time_ids)  # 256 * 6
        + TEXT_ENCODER_2_PROJECTION_DIM  # + 1280
    )
    if passed_add_embed_dim != UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM:
        raise ValueError(
            f"Model expects an added time embedding vector of length {UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids


def get_optimizer(config, trainable_params):
    # "Optimizer to use: AdamW, AdamW8bit, Lion, SGDNesterov, SGDNesterov8bit, PagedAdamW8bit, Lion8bit, PagedLion8bit, DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, DAdaptLion, DAdaptSGD, Adafactor"

    optimizer_type = config.train.optimizer_type
    if optimizer_type is None or optimizer_type == "":
        optimizer_type = "AdamW"
    optimizer_type = optimizer_type.lower()

    # 引数を分解する
    optimizer_kwargs = {}
    if config.train.optimizer_args is not None and len(config.train.optimizer_args) > 0:
        for arg in config.train.optimizer_args:
            key, value = arg.split("=")
            value = ast.literal_eval(value)

            optimizer_kwargs[key] = value

    lr = config.train.lr
    optimizer = None

    if optimizer_type == "Lion".lower():
        try:
            import lion_pytorch
        except ImportError:
            raise ImportError("No lion_pytorch / lion_pytorch がインストールされていないようです")
        print(f"use Lion optimizer | {optimizer_kwargs}")
        optimizer_class = lion_pytorch.Lion
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type.endswith("8bit".lower()):
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsandbytes / bitsandbytesがインストールされていないようです")

        if optimizer_type == "AdamW8bit".lower():
            print(f"use 8-bit AdamW optimizer | {optimizer_kwargs}")
            optimizer_class = bnb.optim.AdamW8bit
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "SGDNesterov8bit".lower():
            print(f"use 8-bit SGD with Nesterov optimizer | {optimizer_kwargs}")
            if "momentum" not in optimizer_kwargs:
                print(
                    f"8-bit SGD with Nesterov must be with momentum, set momentum to 0.9 / 8-bit SGD with Nesterovはmomentum指定が必須のため0.9に設定します"
                )
                optimizer_kwargs["momentum"] = 0.9

            optimizer_class = bnb.optim.SGD8bit
            optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)

        elif optimizer_type == "Lion8bit".lower():
            print(f"use 8-bit Lion optimizer | {optimizer_kwargs}")
            try:
                optimizer_class = bnb.optim.Lion8bit
            except AttributeError:
                raise AttributeError(
                    "No Lion8bit. The version of bitsandbytes installed seems to be old. Please install 0.38.0 or later. / Lion8bitが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.38.0以上をインストールしてください"
                )
        elif optimizer_type == "PagedAdamW8bit".lower():
            print(f"use 8-bit PagedAdamW optimizer | {optimizer_kwargs}")
            try:
                optimizer_class = bnb.optim.PagedAdamW8bit
            except AttributeError:
                raise AttributeError(
                    "No PagedAdamW8bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later. / PagedAdamW8bitが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.39.0以上をインストールしてください"
                )
        elif optimizer_type == "PagedLion8bit".lower():
            print(f"use 8-bit Paged Lion optimizer | {optimizer_kwargs}")
            try:
                optimizer_class = bnb.optim.PagedLion8bit
            except AttributeError:
                raise AttributeError(
                    "No PagedLion8bit. The version of bitsandbytes installed seems to be old. Please install 0.39.0 or later. / PagedLion8bitが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.39.0以上をインストールしてください"
                )

        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "SGDNesterov".lower():
        print(f"use SGD with Nesterov optimizer | {optimizer_kwargs}")
        if "momentum" not in optimizer_kwargs:
            print(f"SGD with Nesterov must be with momentum, set momentum to 0.9 / SGD with Nesterovはmomentum指定が必須のため0.9に設定します")
            optimizer_kwargs["momentum"] = 0.9

        optimizer_class = torch.optim.SGD
        optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)

    elif optimizer_type.startswith("DAdapt".lower()) or optimizer_type == "Prodigy".lower():
        # check lr and lr_count, and print warning
        actual_lr = lr
        lr_count = 1
        if type(trainable_params) == list and type(trainable_params[0]) == dict:
            lrs = set()
            actual_lr = trainable_params[0].get("lr", actual_lr)
            for group in trainable_params:
                lrs.add(group.get("lr", actual_lr))
            lr_count = len(lrs)

        if actual_lr <= 0.1:
            print(
                f"learning rate is too low. If using D-Adaptation or Prodigy, set learning rate around 1.0 / 学習率が低すぎるようです。D-AdaptationまたはProdigyの使用時は1.0前後の値を指定してください: lr={actual_lr}"
            )
            print("recommend option: lr=1.0 / 推奨は1.0です")
        if lr_count > 1:
            print(
                f"when multiple learning rates are specified with dadaptation (e.g. for Text Encoder and U-Net), only the first one will take effect / D-AdaptationまたはProdigyで複数の学習率を指定した場合（Text EncoderとU-Netなど）、最初の学習率のみが有効になります: lr={actual_lr}"
            )

        if optimizer_type.startswith("DAdapt".lower()):
            # DAdaptation family
            # check dadaptation is installed
            try:
                import dadaptation
                import dadaptation.experimental as experimental
            except ImportError:
                raise ImportError("No dadaptation / dadaptation がインストールされていないようです")

            # set optimizer
            if optimizer_type == "DAdaptation".lower() or optimizer_type == "DAdaptAdamPreprint".lower():
                optimizer_class = experimental.DAdaptAdamPreprint
                print(f"use D-Adaptation AdamPreprint optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdaGrad".lower():
                optimizer_class = dadaptation.DAdaptAdaGrad
                print(f"use D-Adaptation AdaGrad optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdam".lower():
                optimizer_class = dadaptation.DAdaptAdam
                print(f"use D-Adaptation Adam optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdan".lower():
                optimizer_class = dadaptation.DAdaptAdan
                print(f"use D-Adaptation Adan optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptAdanIP".lower():
                optimizer_class = experimental.DAdaptAdanIP
                print(f"use D-Adaptation AdanIP optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptLion".lower():
                optimizer_class = dadaptation.DAdaptLion
                print(f"use D-Adaptation Lion optimizer | {optimizer_kwargs}")
            elif optimizer_type == "DAdaptSGD".lower():
                optimizer_class = dadaptation.DAdaptSGD
                print(f"use D-Adaptation SGD optimizer | {optimizer_kwargs}")
            else:
                raise ValueError(f"Unknown optimizer type: {optimizer_type}")

            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
        else:
            # Prodigy
            # check Prodigy is installed
            try:
                import prodigyopt
            except ImportError:
                raise ImportError("No Prodigy / Prodigy がインストールされていないようです")

            print(f"use Prodigy optimizer | {optimizer_kwargs}")
            optimizer_class = prodigyopt.Prodigy
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "Adafactor".lower():
        # 引数を確認して適宜補正する
        if "relative_step" not in optimizer_kwargs:
            optimizer_kwargs["relative_step"] = True  # default
        if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
            print(f"set relative_step to True because warmup_init is True / warmup_initがTrueのためrelative_stepをTrueにします")
            optimizer_kwargs["relative_step"] = True
        print(f"use Adafactor optimizer | {optimizer_kwargs}")

        if optimizer_kwargs["relative_step"]:
            print(f"relative_step is true / relative_stepがtrueです")
            if lr != 0.0:
                print(f"learning rate is used as initial_lr / 指定したlearning rateはinitial_lrとして使用されます")
            config.train.lr = None

            # trainable_paramsがgroupだった時の処理：lrを削除する
            if type(trainable_params) == list and type(trainable_params[0]) == dict:
                has_group_lr = False
                for group in trainable_params:
                    p = group.pop("lr", None)
                    has_group_lr = has_group_lr or (p is not None)

                if has_group_lr:
                    # 一応argsを無効にしておく TODO 依存関係が逆転してるのであまり望ましくない
                    print(f"unet_lr and text_encoder_lr are ignored / unet_lrとtext_encoder_lrは無視されます")
                    config.train.unet_lr = None
                    config.train.text_encoder_lr = None

            if config.train.lr_scheduler != "adafactor":
                print(f"use adafactor_scheduler / スケジューラにadafactor_schedulerを使用します")
            config.train.lr_scheduler = f"adafactor:{lr}"  # ちょっと微妙だけど

            lr = None
        else:
            if config.train.max_grad_norm != 0.0:
                print(
                    f"because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0 / max_grad_normが設定されているためclip_grad_normが有効になります。0に設定して無効にしたほうがいいかもしれません"
                )
            if config.train.lr_scheduler != "constant_with_warmup":
                print(f"constant_with_warmup will be good / スケジューラはconstant_with_warmupが良いかもしれません")
            if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
                print(f"clip_threshold=1.0 will be good / clip_thresholdは1.0が良いかもしれません")

        optimizer_class = transformers.optimization.Adafactor
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "AdamW".lower():
        print(f"use AdamW optimizer | {optimizer_kwargs}")
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    if optimizer is None:
        # 任意のoptimizerを使う
        optimizer_type = config.train.optimizer_type  # lowerでないやつ（微妙）
        print(f"use {optimizer_type} | {optimizer_kwargs}")
        if "." not in optimizer_type:
            optimizer_module = torch.optim
        else:
            values = optimizer_type.split(".")
            optimizer_module = importlib.import_module(".".join(values[:-1]))
            optimizer_type = values[-1]

        optimizer_class = getattr(optimizer_module, optimizer_type)
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
    optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

    return optimizer_name, optimizer_args, optimizer
    

def get_scheduler_fix(config, optimizer: Optimizer, num_processes: int = 1):
    """
    Unified API to get any scheduler from its name.
    """
    name = config.train.lr_scheduler
    num_warmup_steps: Optional[int] = config.train.lr_warmup_steps
    num_training_steps = config.train.iterations * num_processes  # * args.gradient_accumulation_steps
    num_cycles = config.train.lr_scheduler_num_cycles
    power = config.train.lr_scheduler_power

    lr_scheduler_kwargs = {}  # get custom lr_scheduler kwargs
    if config.train.lr_scheduler_args is not None and len(config.train.lr_scheduler_args) > 0:
        for arg in config.train.lr_scheduler_args:
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            lr_scheduler_kwargs[key] = value

    def wrap_check_needless_num_warmup_steps(return_vals):
        if num_warmup_steps is not None and num_warmup_steps != 0:
            raise ValueError(f"{name} does not require `num_warmup_steps`. Set None or 0.")
        return return_vals

    if name.startswith("adafactor"):
        assert (
            type(optimizer) == transformers.optimization.Adafactor
        ), f"adafactor scheduler must be used with Adafactor optimizer / adafactor schedulerはAdafactorオプティマイザと同時に使ってください"
        initial_lr = float(name.split(":")[1])
        # print("adafactor scheduler init lr", initial_lr)
        return wrap_check_needless_num_warmup_steps(transformers.optimization.AdafactorSchedule(optimizer, initial_lr))

    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if name == SchedulerType.CONSTANT:
        return wrap_check_needless_num_warmup_steps(schedule_func(optimizer, **lr_scheduler_kwargs))

    if name == SchedulerType.PIECEWISE_CONSTANT:
        return schedule_func(optimizer, **lr_scheduler_kwargs)  # step_rules and last_epoch are given as kwargs

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **lr_scheduler_kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            **lr_scheduler_kwargs,
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, power=power, **lr_scheduler_kwargs
        )

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **lr_scheduler_kwargs)


def get_random_resolution_in_bucket(bucket_resolution: int = 512) -> tuple[int, int]:
    max_resolution = bucket_resolution
    min_resolution = bucket_resolution // 2

    step = 64

    min_step = min_resolution // step
    max_step = max_resolution // step

    height = torch.randint(min_step, max_step, (1,)).item() * step
    width = torch.randint(min_step, max_step, (1,)).item() * step

    return height, width

def text2img(pipe: DiffusionPipeline,
             prompts: Union[str, list[str]], 
             negative_prompt: Union[str, list[str]] = "", 
             width: int = 512, 
             height: int = 512,
             num_inference_steps: int = 30,
             guidance_scale: int = 7.5,
             seed: int = None,
             generate_num: int = 1,
             tag: str = "",
             **kwargs):
    # to avoid CUDA-OOM, generate images prompt-by-prompt, unless generate_num is 1
    
    samples = []
    
    if generate_num == 1:
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * len(prompts)
        images = pipe(
            prompts,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=generate_num,
            generator=torch.manual_seed(seed) if seed is not None else None,
        ).images
        texts = [f"sample/{prompt.replace(' ', '_')}{'(' + tag + ')' if tag else ''}" for prompt in prompts]
        samples = list(zip(texts, images))
    else:
        for prompt in prompts:
            images = pipe(
                prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=generate_num,
                generator=torch.manual_seed(seed) if seed is not None else None,
            ).images
            texts = [f"sample/{prompt.replace(' ', '_')}({tag}{', ' if tag else ''}{i})" for i in range(generate_num)]
            samples.extend(list(zip(texts, images)))
    
    return samples

@torch.no_grad()
def latent2img(pipe: DiffusionPipeline,
               scheduler,
               noise_pred: torch.FloatTensor,
               latents: torch.FloatTensor,
               timestep: int,
               tag: str = "ori",
               **kwargs):
    
    noise_pred = noise_pred.cuda()
    latents = latents.cuda()
    latents = scheduler.step(noise_pred, timestep, latents).prev_sample
    imgs = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    do_denormalize = [True] * imgs.shape[0]
    imgs = pipe.image_processor.postprocess(imgs, do_denormalize=do_denormalize)
    samples = [(f"sample/latent({tag}, {i})", img) for i, img in enumerate(imgs)]
    
    return samples
    