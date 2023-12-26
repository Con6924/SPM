import argparse
import gc
from pathlib import Path

import torch
from typing import Literal

from src.configs.generation_config import load_config_from_yaml, GenerationConfig
from src.configs.config import parse_precision
from src.engine import train_util
from src.models import model_util
from src.models.spm import SPMLayer, SPMNetwork
from src.models.merge_spm import load_state_dict

DEVICE_CUDA = torch.device("cuda:0")
UNET_NAME = "unet"
TEXT_ENCODER_NAME = "text_encoder"
MATCHING_METRICS = Literal[
    "clipcos",
    "clipcos_tokenuni",
    "tokenuni",
]

def flush():
    torch.cuda.empty_cache()
    gc.collect()

def calculate_matching_score(
        prompt_tokens,
        prompt_embeds, 
        erased_prompt_tokens, 
        erased_prompt_embeds, 
        matching_metric: MATCHING_METRICS,
        special_token_ids: set[int],
        weight_dtype: torch.dtype = torch.float32,
    ):
    scores = []
    if "clipcos" in matching_metric:
        clipcos = torch.cosine_similarity(
                    prompt_embeds.flatten(1, 2), 
                    erased_prompt_embeds.flatten(1, 2), 
                    dim=-1).cpu()
        scores.append(clipcos)
    if "tokenuni" in matching_metric:
        prompt_set = set(prompt_tokens[0].tolist()) - special_token_ids
        tokenuni = []
        for ep in erased_prompt_tokens:
            ep_set = set(ep.tolist()) - special_token_ids
            tokenuni.append(len(prompt_set.intersection(ep_set)) / len(ep_set))
        scores.append(torch.tensor(tokenuni).to("cpu", dtype=weight_dtype))
    return torch.max(torch.stack(scores), dim=0)[0]

def infer_with_spm(
        spm_paths: list[str],
        config: GenerationConfig,
        matching_metric: MATCHING_METRICS,
        assigned_multipliers: list[float] = None,
        base_model: str = "CompVis/stable-diffusion-v1-4",
        v2: bool = False,
        precision: str = "fp32",
    ):

    spm_model_paths = [lp / f"{lp.name}_last.safetensors" if lp.is_dir() else lp for lp in spm_paths]

    weight_dtype = parse_precision(precision)
    
    # load the pretrained SD
    tokenizer, text_encoder, unet, pipe = model_util.load_checkpoint_model(
        base_model,
        v2=v2,
        weight_dtype=weight_dtype
    )
    special_token_ids = set(tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values()))

    text_encoder.to(DEVICE_CUDA, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(DEVICE_CUDA, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()

    # load the SPM modules
    spms, metadatas = zip(*[
        load_state_dict(spm_model_path, weight_dtype) for spm_model_path in spm_model_paths
    ])
    # check if SPMs are compatible
    assert all([metadata["rank"] == metadatas[0]["rank"] for metadata in metadatas])

    # get the erased concept
    erased_prompts = [md["prompts"].split(",") for md in metadatas]
    erased_prompts_count = [len(ep) for ep in erased_prompts]
    print(f"Erased prompts: {erased_prompts}")

    erased_prompts_flatten = [item for sublist in erased_prompts for item in sublist]
    erased_prompt_embeds, erased_prompt_tokens = train_util.encode_prompts(
        tokenizer, text_encoder, erased_prompts_flatten, return_tokens=True
        )

    network = SPMNetwork(
        unet,
        rank=int(float(metadatas[0]["rank"])),
        alpha=float(metadatas[0]["alpha"]),
        module=SPMLayer,
    ).to(DEVICE_CUDA, dtype=weight_dtype)

    with torch.no_grad():
        for prompt in config.prompts:
            prompt += config.unconditional_prompt
            print(f"Generating for prompt: {prompt}")
            prompt_embeds, prompt_tokens = train_util.encode_prompts(
                tokenizer, text_encoder, [prompt], return_tokens=True
                )
            if assigned_multipliers is not None:
                multipliers = torch.tensor(assigned_multipliers).to("cpu", dtype=weight_dtype)
                if assigned_multipliers == [0,0,0]:
                    matching_metric = "aazeros"
                elif assigned_multipliers == [1,1,1]:
                    matching_metric = "zzone"
            else:
                multipliers = calculate_matching_score(
                    prompt_tokens,
                    prompt_embeds, 
                    erased_prompt_tokens, 
                    erased_prompt_embeds, 
                    matching_metric=matching_metric,
                    special_token_ids=special_token_ids,
                    weight_dtype=weight_dtype
                    )
                multipliers = torch.split(multipliers, erased_prompts_count)
            print(f"multipliers: {multipliers}")
            weighted_spm = dict.fromkeys(spms[0].keys())
            used_multipliers = []
            for spm, multiplier in zip(spms, multipliers):
                max_multiplier = torch.max(multiplier)
                for key, value in spm.items():
                    if weighted_spm[key] is None:
                        weighted_spm[key] = value * max_multiplier
                    else:
                        weighted_spm[key] += value * max_multiplier
                used_multipliers.append(max_multiplier.item())
            network.load_state_dict(weighted_spm)
            with network:
                images = pipe(
                    negative_prompt=config.negative_prompt,
                    width=config.width,
                    height=config.height,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    generator=torch.cuda.manual_seed(config.seed),
                    num_images_per_prompt=config.generate_num,
                    prompt_embeds=prompt_embeds,
                ).images
            folder = Path(config.save_path.format(prompt.replace(" ", "_"), "0")).parent
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)
            for i, image in enumerate(images):
                image.save(
                    config.save_path.format(
                        prompt.replace(" ", "_"), i
                    )
                )

def main(args):
    spm_path = [Path(lp) for lp in args.spm_path]
    generation_config = load_config_from_yaml(args.config)
            
    infer_with_spm(
        spm_path,
        generation_config,
        args.matching_metric,
        assigned_multipliers=args.spm_multiplier,
        base_model=args.base_model,
        v2=args.v2,
        precision=args.precision,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/generation.yaml",
        help="Base configs for image generation.",
    )
    parser.add_argument(
        "--spm_path",
        required=True,
        nargs="*",
        help="SPM(s) to use.",
    )
    parser.add_argument(
        "--spm_multiplier",
        nargs="*",
        type=float,
        default=None,
        help="Assign multipliers for SPM model or set to `None` to use Facilitated Transport.",
    )
    parser.add_argument(
        "--matching_metric",
        type=str,
        default="clipcos_tokenuni",
        help="matching metric for prompt vs erased concept",
    )
    
    # model configs
    parser.add_argument(
        "--base_model",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Base model for generation.",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use the 2.x version of the SD.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Precision for the base model.",
    )

    args = parser.parse_args()

    main(args)
