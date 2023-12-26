import argparse
import gc
from pathlib import Path
from typing import Literal
import warnings

import torch
from torch.utils.data import DataLoader
from accelerate import PartialState, Accelerator

from src.configs import config
from src.configs.config import RootConfig
from src.configs.generation_config import GenerationConfig
from src.engine import train_util
from src.evaluation import *
from src.models import model_util
from src.models.spm import SPMLayer, SPMNetwork
from src.models.merge_spm import load_state_dict
from src.misc.sld_pipeline import SLDPipeline

DIFFUSERS_CACHE_DIR = ".cache/" 
UNET_NAME = "unet"
TEXT_ENCODER_NAME = "text_encoder"
MATCHING_METRICS = Literal[
    "clipcos",
    "clipcos_tokenuni",
    "tokenuni",
    "allone",
]
distributed_state = PartialState()
accelerator = Accelerator()


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def parse_extra_args(extra_args):
    if extra_args is None or extra_args == ['']:
        return {}
    extra_args_dict = {}
    for extra_arg in extra_args:
        key, value = extra_arg.split("=")
        # convert value to various types
        if value.isdigit():
            value = int(value)
        elif value.replace(".", "", 1).isdigit():
            value = float(value)
        elif value[0] == "[" and value[-1] == "]":
            value = [i.replace('+', ' ') for i in value[1:-1].split(",")]
            value = [v.strip() for v in value]
            if value[0].isdigit():
                value = [int(v) for v in value]
            elif value[0].replace(".", "", 1).isdigit():
                value = [float(v) for v in value]
        extra_args_dict[key] = value
    return extra_args_dict


def get_dataloader(args, num_processes=1):
    # parse task_args arguments
    task_args = parse_extra_args(args.task_args)
    task_args["save_folder"] = args.img_save_path
    task_args["output_path"] = args.save_path

    # parse generation arguments
    cfg = parse_extra_args(args.generation_cfg)
    cfg = GenerationConfig(**cfg)

    dataset_class = None
    if args.task == "general":
        dataset_class = ClipTemplateDataset
    elif args.task == "artwork":
        dataset_class = ArtworkDataset
    elif args.task == "i2p":
        dataset_class = I2PDataset
    elif args.task == "coco":
        dataset_class = Coco30kGenerationDataset
    else:
        raise ValueError(f"Unknown task: {args.task}")
    dataset = dataset_class(**task_args, base_cfg=cfg)
    dataloader = DataLoader(dataset, batch_size=num_processes, num_workers=0, shuffle=False)
    return dataloader


def get_evaluator(args):
    evaluator_class = None
    if args.task == "general":
        evaluator_class = ClipEvaluator
    elif args.task == "artwork":
        evaluator_class = ArtworkEvaluator
    elif args.task == "i2p":
        evaluator_class = I2PEvaluator
    elif args.task == "coco":
        evaluator_class = CocoEvaluator
    else:
        raise ValueError(f"Unknown task: {args.task}")
    evaluator = evaluator_class(
        save_folder=args.img_save_path, output_path=args.save_path
    )
    return evaluator


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
    if "allone" in matching_metric:
        scores.append(torch.ones(prompt_embeds.shape[0]).to("cpu", dtype=weight_dtype))
    if "clipcos" in matching_metric:
        clipcos = torch.cosine_similarity(
            prompt_embeds.flatten(1, 2), erased_prompt_embeds.flatten(1, 2), dim=-1
        ).cpu()
        scores.append(clipcos)
    if "tokenuni" in matching_metric:
        prompt_set = set(prompt_tokens[0].tolist()) - special_token_ids
        tokenuni = []
        for ep in erased_prompt_tokens:
            ep_set = set(ep.tolist()) - special_token_ids
            tokenuni.append(len(prompt_set.intersection(ep_set)) / len(ep_set))
        scores.append(torch.tensor(tokenuni).to("cpu", dtype=weight_dtype))
    return torch.max(torch.stack(scores), dim=0)[0]


@torch.no_grad()
def infer_with_spm(
    dataloader: DataLoader,
    spm_paths: list[str],
    matching_metric: MATCHING_METRICS,
    facilitate_factor: float = 1.0,
    assigned_multipliers: list[float] = None,
    finetuned_model_path: str = None,
    sld_target_concept: str = None,
    base_model: str = "CompVis/stable-diffusion-v1-4",
    v2: bool = False,
    precision: str = "fp32",
):
    spm_model_paths = [
        lp / f"{lp.name}_last.safetensors" if lp.is_dir() else lp for lp in spm_paths
    ]

    weight_dtype = config.parse_precision(precision)

    if finetuned_model_path is not None and Path(finetuned_model_path).is_dir():
        # folder path for the diffuser model
        base_model = finetuned_model_path
        print(f"Using models from {base_model}")

    # load the pretrained SD
    tokenizer, text_encoder, unet, pipe = model_util.load_checkpoint_model(
        base_model,
        v2=v2,
        weight_dtype=weight_dtype,
        device=distributed_state.device,
    )
    special_token_ids = set(
        tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
    )

    text_encoder.to(distributed_state.device, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(distributed_state.device, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()

    if len(spm_model_paths) > 0:
        # load the SPM models
        spms, metadatas = zip(
            *[
                load_state_dict(spm_model_path, weight_dtype)
                for spm_model_path in spm_model_paths
            ]
        )
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

        # create the SPM network
        network = SPMNetwork(
            unet,
            rank=int(float(metadatas[0]["rank"])),
            alpha=float(metadatas[0]["alpha"]),
            module=SPMLayer,
        ).to(distributed_state.device, dtype=weight_dtype)
    if finetuned_model_path is not None:
        if finetuned_model_path.endswith('.bin'):
            # concept-ablation
            st = torch.load(finetuned_model_path, map_location='cpu')
            if 'text_encoder' in st:
                text_encoder.load_state_dict(st['text_encoder'])
            for name, params in unet.named_parameters():
                if name in st['unet']:
                    params.data.copy_(st['unet'][f'{name}'])
        elif finetuned_model_path.endswith('.pt'):
            # ESD
            unet.load_state_dict(torch.load(finetuned_model_path, map_location='cpu'))
        elif Path(finetuned_model_path).is_dir():
            # SA
            pass
        elif finetuned_model_path.lower() == 'sld':
            # SLD
            pipe = SLDPipeline.from_pretrained(
                base_model,
                safety_checker=None,
                cache_dir=DIFFUSERS_CACHE_DIR, 
                torch_dtype=weight_dtype,
            ).to(distributed_state.device)
            if sld_target_concept.lower() != 'i2p':
                pipe.safety_concept = sld_target_concept
            print(f"Using SLD to erase target concept: {pipe.safety_concept}")
    if len(spm_model_paths) == 0 and finetuned_model_path is None:
        warnings.warn("No SPM model or finetuned model is provided, using the pretrained model directly.")

    print("Generating images...")
    with distributed_state.split_between_processes(dataloader.dataset.data) as dataset:
        dataset = tqdm(dataset) if distributed_state.is_main_process else dataset
        for cfg in dataset:
            # save path checking
            folder = Path(cfg['save_path']).parent
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)

            # check whether the image has been generated
            if cfg['generate_num'] > 1:
                if all([Path(cfg['save_path'].format(idx)).exists() for idx in range(cfg['generate_num'])]):
                    print(f"Skipping {cfg['save_path']}, already exists.")
                    continue
            else:
                if Path(cfg['save_path']).exists():
                    print(f"Skipping {cfg['save_path']}, already exists.")
                    continue

            prompts = [p + ", " + cfg['unconditional_prompt'] if cfg['unconditional_prompt'] else p for p in cfg['prompts']]
            
            # generate images
            if len(spm_model_paths) > 0:
                prompt_embeds, prompt_tokens = train_util.encode_prompts(
                    tokenizer, text_encoder, prompts, return_tokens=True
                )
                if assigned_multipliers is not None:
                    multipliers = torch.tensor(assigned_multipliers).to(
                        "cpu", dtype=weight_dtype
                    )
                    matching_metric = "_".join([str(i) for i in assigned_multipliers])
                else:
                    multipliers = calculate_matching_score(
                        prompt_tokens,
                        prompt_embeds,
                        erased_prompt_tokens,
                        erased_prompt_embeds,
                        matching_metric=matching_metric,
                        special_token_ids=special_token_ids,
                        weight_dtype=weight_dtype,
                    )
                    multipliers = torch.split(multipliers, erased_prompts_count)
                print(f"multipliers: {multipliers}")
                weighted_spm = dict.fromkeys(spms[0].keys())
                used_multipliers = []
                for spm, multiplier in zip(spms, multipliers):
                    max_multiplier = torch.max(multiplier)
                    max_multiplier *= facilitate_factor
                    for key, value in spm.items():
                        if weighted_spm[key] is None:
                            weighted_spm[key] = value * max_multiplier
                        else:
                            weighted_spm[key] += value * max_multiplier
                    used_multipliers.append(max_multiplier.item())
                network.load_state_dict(weighted_spm)

                with network:
                    images = pipe(
                        negative_prompt=cfg['negative_prompt'],
                        width=cfg['width'],
                        height=cfg['height'],
                        num_inference_steps=cfg['num_inference_steps'],
                        guidance_scale=cfg['guidance_scale'],
                        generator=torch.cuda.manual_seed(cfg['seed']) if cfg['seed'] else None,
                        num_images_per_prompt=cfg['generate_num'],
                        prompt_embeds=prompt_embeds,
                    ).images
            elif sld_target_concept:
                images = pipe(
                    prompt=prompts, 
                    negative_prompt=[cfg['negative_prompt']]*len(prompts),
                    width=cfg['width'],
                    height=cfg['height'],
                    num_inference_steps=cfg['num_inference_steps'],
                    guidance_scale=cfg['guidance_scale'],
                    generator=torch.cuda.manual_seed(cfg['seed']) if cfg['seed'] else None,
                    num_images_per_prompt=cfg['generate_num'],
                    sld_warmup_steps=7,
                    sld_guidance_scale=2000,
                    sld_threshold=0.025,
                    sld_momentum_scale=0.5,
                    sld_mom_beta=0.7
                ).images
            else:
                images = pipe(
                    prompt=prompts, 
                    negative_prompt=[cfg['negative_prompt']]*len(prompts),
                    width=cfg['width'],
                    height=cfg['height'],
                    num_inference_steps=cfg['num_inference_steps'],
                    guidance_scale=cfg['guidance_scale'],
                    generator=torch.cuda.manual_seed(cfg['seed']) if cfg['seed'] else None,
                    num_images_per_prompt=cfg['generate_num'],
                ).images

            # save generated images
            if len(images) > 1:
                for idx, image in enumerate(images):
                    image.save(cfg['save_path'].format(idx))
            else:
                images[0].save(cfg['save_path'])


def main(args):
    # data preparation
    spm_paths = [Path(lp) for lp in args.spm_paths] if args.spm_paths else []

    print(f"Using {distributed_state.num_processes} processes for evaluation.")
    dataloader = get_dataloader(args, num_processes=distributed_state.num_processes)

    # inference
    if not args.eval_only:
        infer_with_spm(
            dataloader,
            spm_paths=spm_paths,
            matching_metric=args.matching_metric,
            facilitate_factor=args.facilitate_factor,
            assigned_multipliers=args.spm_multiplier,
            finetuned_model_path=args.ft_model_path,
            sld_target_concept=args.sld_target_concept,
            base_model=args.base_model,
            v2=args.v2,
            precision=args.precision,
        )
        accelerator.wait_for_everyone()

    # evaluation
    if not args.gen_only and distributed_state.is_main_process:
        evaluator = get_evaluator(args)
        evaluator.evaluation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        choices=["general", "artwork", "i2p", "coco"],
        help="Task to evaluate.",
    )
    parser.add_argument(
        "--task_args",
        nargs="*",
        help="""Extra arguments for the task. Acceptable arguments:
            task=general: concepts(list[str]), num_templates(optional, int, default=80), num_images_per_template(optional, int, default=10);
            task=artwork: datasets(list[str]);
            task=i2p: None.
            task=coco: None.
        """,
    )
    parser.add_argument(
        "--generation_cfg",
        nargs="*",
        help="Arguments to overwrite default generation configs.",
    )
    parser.add_argument(
        "--img_save_path",
        type=str,
        required=True,
        help="Path to save generated images.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save evaluation results.",
    )

    parser.add_argument(
        "--spm_paths",
        nargs="*",
        help="SPM(s) to use.",
    )
    parser.add_argument(
        "--spm_multiplier",
        nargs="*",
        type=float,
        default=None,
        help="Assign multipliers for SPM model or use Facilitated Transport.",
    )
    parser.add_argument(
        "--facilitate_factor",
        type=float,
        default=1.0,
        help="Facilitate factor for SPM.",
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

    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip generation and only evaluate the generated images.",
    )

    parser.add_argument(
        "--gen_only",
        action="store_true",
        help="Skip evaluation and only generate images.",
    )

    # for comparative study
    
    parser.add_argument(
        "--ft_model_path",
        default=None,
        help=".pt for ESD, .bin for ConAbl, and SLD for SLD.",
    )
    parser.add_argument(
        "--sld_target_concept",
        default=None,
        help="SLD requires named targets for erasing.",
    )

    args = parser.parse_args()
    
    if args.ft_model_path and args.ft_model_path.lower() == 'sld' and args.sld_target_concept is None:
        raise ValueError("SLD requires named targets for erasing.")

    main(args)
