# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py
# - https://github.com/p1atdev/LECO/blob/main/train_lora_xl.py

import argparse
from pathlib import Path
import gc

import torch
from tqdm import tqdm

from src.models.spm import (
    SPMNetwork,
    SPMLayer,
)
from src.engine.sampling import sample_xl
import src.engine.train_util as train_util
from src.models import model_util
from src.evaluation import eval_util
from src.configs import config as config_pkg
from src.configs import prompt as prompt_pkg
from src.configs.config import RootConfig
from src.configs.prompt import PromptEmbedsCache, PromptEmbedsPair, PromptSettings, PromptEmbedsXL

import wandb

DEVICE_CUDA = torch.device("cuda:0")
NUM_IMAGES_PER_PROMPT = 1


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def train(
    config: RootConfig,
    prompts: list[PromptSettings],
):
    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
        "config": config.json(),
    }
    model_metadata = {
        "prompts": ",".join([prompt.target for prompt in prompts]),
        "rank": str(config.network.rank),
        "alpha": str(config.network.alpha),
    }
    save_path = Path(config.save.path)

    if config.logging.verbose:
        print(metadata)

    weight_dtype = config_pkg.parse_precision(config.train.precision)
    save_weight_dtype = config_pkg.parse_precision(config.train.precision)

    if config.logging.use_wandb:
        wandb.init(project=f"SPM", 
                   config=metadata, 
                   name=config.logging.run_name, 
                   settings=wandb.Settings(symlink=False))

    (
        tokenizers,
        text_encoders,
        unet,
        noise_scheduler,
        pipe
    ) = model_util.load_models_xl(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
    )

    for text_encoder in text_encoders:
        text_encoder.to(DEVICE_CUDA, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    unet.to(DEVICE_CUDA, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()

    network = SPMNetwork(
        unet,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        module=SPMLayer,
    ).to(DEVICE_CUDA, dtype=weight_dtype)

    trainable_params = network.prepare_optimizer_params(
        config.train.text_encoder_lr, config.train.unet_lr, config.train.lr
    )
    optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(
        config, trainable_params
    )
    lr_scheduler = train_util.get_scheduler_fix(config, optimizer)
    criteria = torch.nn.MSELoss()

    print("Prompts")
    for settings in prompts:
        print(settings)

    cache = PromptEmbedsCache()
    prompt_pairs: list[PromptEmbedsPair] = []

    with torch.no_grad():
        for settings in prompts:
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                if cache[prompt] == None:
                    cache[prompt] = PromptEmbedsXL(
                        train_util.encode_prompts_xl(
                            tokenizers,
                            text_encoders,
                            [prompt],
                            num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
                        )
                    )

            prompt_pair = PromptEmbedsPair(
                criteria,
                cache[settings.target],
                cache[settings.positive],
                cache[settings.unconditional],
                cache[settings.neutral],
                settings,
            )
            assert prompt_pair.sampling_batch_size % prompt_pair.batch_size == 0
            prompt_pairs.append(prompt_pair)

    flush()

    pbar = tqdm(range(config.train.iterations))
    loss = None

    for i in pbar:
        with torch.no_grad():
            noise_scheduler.set_timesteps(
                config.train.max_denoising_steps, device=DEVICE_CUDA
            )

            optimizer.zero_grad()

            prompt_pair: PromptEmbedsPair = prompt_pairs[
                torch.randint(0, len(prompt_pairs), (1,)).item()
            ]

            timesteps_to = torch.randint(
                1, config.train.max_denoising_steps, (1,)
            ).item()

            height, width = (
                prompt_pair.resolution,
                prompt_pair.resolution,
            )
            if prompt_pair.dynamic_resolution:
                height, width = train_util.get_random_resolution_in_bucket(
                    prompt_pair.resolution
                )

            if config.logging.verbose:
                print("guidance_scale:", prompt_pair.guidance_scale)
                print("resolution:", prompt_pair.resolution)
                print("dynamic_resolution:", prompt_pair.dynamic_resolution)
                if prompt_pair.dynamic_resolution:
                    print("bucketed resolution:", (height, width))
                print("batch_size:", prompt_pair.batch_size)
                print("dynamic_crops:", prompt_pair.dynamic_crops)

            latents = train_util.get_initial_latents(
                noise_scheduler, prompt_pair.batch_size, height, width, 1
            ).to(DEVICE_CUDA, dtype=weight_dtype)

            add_time_ids = train_util.get_add_time_ids(
                height,
                width,
                dynamic_crops=prompt_pair.dynamic_crops,
                dtype=weight_dtype,
            ).to(DEVICE_CUDA, dtype=weight_dtype)

            with network:
                denoised_latents = train_util.diffusion_xl(
                    unet,
                    noise_scheduler,
                    latents,
                    text_embeddings=train_util.concat_embeddings(
                        prompt_pair.unconditional.text_embeds,
                        prompt_pair.target.text_embeds,
                        prompt_pair.batch_size,
                    ),
                    add_text_embeddings=train_util.concat_embeddings(
                        prompt_pair.unconditional.pooled_embeds,
                        prompt_pair.target.pooled_embeds,
                        prompt_pair.batch_size,
                    ),
                    add_time_ids=train_util.concat_embeddings(
                        add_time_ids, add_time_ids, prompt_pair.batch_size
                    ),
                    start_timesteps=0,
                    total_timesteps=timesteps_to,
                    guidance_scale=3,
                )

            noise_scheduler.set_timesteps(1000)

            current_timestep = noise_scheduler.timesteps[
                int(timesteps_to * 1000 / config.train.max_denoising_steps)
            ]

            positive_latents = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.text_embeds,
                    prompt_pair.positive.text_embeds,
                    prompt_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.pooled_embeds,
                    prompt_pair.positive.pooled_embeds,
                    prompt_pair.batch_size,
                ),
                add_time_ids=train_util.concat_embeddings(
                    add_time_ids, add_time_ids, prompt_pair.batch_size
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)
            neutral_latents = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.text_embeds,
                    prompt_pair.neutral.text_embeds,
                    prompt_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.pooled_embeds,
                    prompt_pair.neutral.pooled_embeds,
                    prompt_pair.batch_size,
                ),
                add_time_ids=train_util.concat_embeddings(
                    add_time_ids, add_time_ids, prompt_pair.batch_size
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)

        with network:
            target_latents = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.text_embeds,
                    prompt_pair.target.text_embeds,
                    prompt_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.pooled_embeds,
                    prompt_pair.target.pooled_embeds,
                    prompt_pair.batch_size,
                ),
                add_time_ids=train_util.concat_embeddings(
                    add_time_ids, add_time_ids, prompt_pair.batch_size
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)

        # ------------------------- latent anchoring part -----------------------------

        if prompt_pair.action == "erase_with_la":
            # noise sampling
            anchors_text, anchors_pool = sample_xl(prompt_pair, tokenizers=tokenizers, text_encoders=text_encoders)

            # get latents
            repeat = prompt_pair.sampling_batch_size // prompt_pair.batch_size
            # TODO: target or positive?
            with network:
                anchor_latents = train_util.predict_noise_xl(
                    unet,
                    noise_scheduler,
                    current_timestep,
                    denoised_latents.repeat(repeat, 1, 1, 1),
                    text_embeddings=anchors_text,
                    add_text_embeddings=anchors_pool,
                    add_time_ids=train_util.concat_embeddings(
                        add_time_ids, add_time_ids, prompt_pair.sampling_batch_size
                    ),
                    guidance_scale=1,
                ).to("cpu", dtype=torch.float32)

            with torch.no_grad():
                anchor_latents_ori = train_util.predict_noise_xl(
                    unet,
                    noise_scheduler,
                    current_timestep,
                    denoised_latents.repeat(repeat, 1, 1, 1),
                    text_embeddings=anchors_text,
                    add_text_embeddings=anchors_pool,
                    add_time_ids=train_util.concat_embeddings(
                        add_time_ids, add_time_ids, prompt_pair.sampling_batch_size
                    ),
                    guidance_scale=1,
                ).to("cpu", dtype=torch.float32)
            anchor_latents_ori.requires_grad_ = False

        else:
            anchor_latents = None
            anchor_latents_ori = None

        # ----------------------------------------------------------------

        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False

        loss = prompt_pair.loss(
            target_latents=target_latents,
            positive_latents=positive_latents,
            neutral_latents=neutral_latents,
            anchor_latents=anchor_latents,
            anchor_latents_ori=anchor_latents_ori,
        )

        loss["loss"].backward()
        if config.train.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                trainable_params, config.train.max_grad_norm, norm_type=2
            )
        optimizer.step()
        lr_scheduler.step()

        pbar.set_description(f"Loss*1k: {loss['loss'].item()*1000:.4f}")

        # logging
        if config.logging.use_wandb:
            log_dict = {"iteration": i}
            loss = {k: v.detach().cpu().item() for k, v in loss.items()}
            log_dict.update(loss)
            lrs = lr_scheduler.get_last_lr()
            if len(lrs) == 1:
                log_dict["lr"] = float(lrs[0])
            else:
                log_dict["lr/textencoder"] = float(lrs[0])
                log_dict["lr/unet"] = float(lrs[-1])
            if config.train.optimizer_type.lower().startswith("dadapt"):
                log_dict["lr/d*lr"] = (
                    optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
                )

            # generate sample images
            if config.logging.interval > 0 and (
                i % config.logging.interval == 0 or i == config.train.iterations - 1
            ):
                print("Generating samples...")
                with network:
                    samples = train_util.text2img(
                        pipe,
                        prompts=config.logging.prompts,
                        negative_prompt=config.logging.negative_prompt,
                        width=config.logging.width,
                        height=config.logging.height,
                        num_inference_steps=config.logging.num_inference_steps,
                        guidance_scale=config.logging.guidance_scale,
                        generate_num=config.logging.generate_num,
                        seed=config.logging.seed,
                    )
                for text, img in samples:
                    log_dict[text] = wandb.Image(img)

                # evaluate on the generated images
                print("Evaluating CLIPScore and CLIPAccuracy...")
                with network:
                    clip_scores, clip_accs = eval_util.clip_eval(pipe, config)
                    for prompt, clip_score, clip_accuracy in zip(
                        config.logging.prompts, clip_scores, clip_accs
                    ):
                        log_dict[f"CLIPScore/{prompt}"] = clip_score
                        log_dict[f"CLIPAccuracy/{prompt}"] = clip_accuracy
                    log_dict[f"CLIPScore/average"] = sum(clip_scores) / len(clip_scores)
                    log_dict[f"CLIPAccuracy/average"] = sum(clip_accs) / len(clip_accs)

            wandb.log(log_dict)

        # save model
        if (
            i % config.save.per_steps == 0
            and i != 0
            and i != config.train.iterations - 1
        ):
            print("Saving...")
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_{i}steps.safetensors",
                dtype=save_weight_dtype,
                metadata=model_metadata,
            )

        del (
            positive_latents,
            neutral_latents,
            target_latents,
            latents,
            anchor_latents,
            anchor_latents_ori,
        )
        flush()

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_last.safetensors",
        dtype=save_weight_dtype,
        metadata=model_metadata,
    )

    del (
        unet,
        noise_scheduler,
        loss,
        optimizer,
        network,
    )

    flush()

    print("Done.")


def main(args):
    config_file = args.config_file

    config = config_pkg.load_config_from_yaml(config_file)
    prompts = prompt_pkg.load_prompts_from_yaml(config.prompts_file)

    train(config, prompts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=True,
        help="Config file for training.",
    )

    args = parser.parse_args()

    main(args)
