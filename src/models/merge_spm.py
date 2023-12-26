# modify from:
# - https://github.com/bmaltais/kohya_ss/blob/master/networks/merge_lora.py

import math
import argparse
import os
import torch
import safetensors
from safetensors.torch import load_file
from diffusers import DiffusionPipeline


def load_state_dict(file_name, dtype):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)

    return sd, metadata


def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    """r
    This method locks the file. see https://github.com/huggingface/safetensors/issues/164
    If the file isn't .safetensors or doesn't have metadata, return empty dict.
    """
    if os.path.splitext(safetensors_file)[1] != ".safetensors":
        return {}

    with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata


def merge_lora_models(models, ratios, merge_dtype):
    base_alphas = {}  # alpha for merged model
    base_dims = {}

    merged_sd = {}
    for model, ratio in zip(models, ratios):
        print(f"loading: {model}")
        lora_sd, lora_metadata = load_state_dict(model, merge_dtype)

        # get alpha and dim
        alphas = {}  # alpha for current model
        dims = {}  # dims for current model
        for key in lora_sd.keys():
            if "alpha" in key:
                lora_module_name = key[: key.rfind(".alpha")]
                alpha = float(lora_sd[key].detach().numpy())
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha
            elif "lora_down" in key:
                lora_module_name = key[: key.rfind(".lora_down")]
                dim = lora_sd[key].size()[0]
                dims[lora_module_name] = dim
                if lora_module_name not in base_dims:
                    base_dims[lora_module_name] = dim

        for lora_module_name in dims.keys():
            if lora_module_name not in alphas:
                alpha = dims[lora_module_name]
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha

        print(f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}")

        # merge
        print(f"merging...")
        for key in lora_sd.keys():
            if "alpha" in key:
                continue

            lora_module_name = key[: key.rfind(".lora_")]

            base_alpha = base_alphas[lora_module_name]
            alpha = alphas[lora_module_name]

            scale = math.sqrt(alpha / base_alpha) * ratio

            if key in merged_sd:
                assert (
                    merged_sd[key].size() == lora_sd[key].size()
                ), f"weights shape mismatch merging v1 and v2, different dims? / 重みのサイズが合いません。v1とv2、または次元数の異なるモデルはマージできません"
                merged_sd[key] = merged_sd[key] + lora_sd[key] * scale
            else:
                merged_sd[key] = lora_sd[key] * scale

    # set alpha to sd
    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)

    print("merged model")
    print(f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}")

    # check all dims are same
    dims_list = list(set(base_dims.values()))
    alphas_list = list(set(base_alphas.values()))
    all_same_dims = True
    all_same_alphas = True
    for dims in dims_list:
        if dims != dims_list[0]:
            all_same_dims = False
            break
    for alphas in alphas_list:
        if alphas != alphas_list[0]:
            all_same_alphas = False
            break

    # build minimum metadata
    dims = f"{dims_list[0]}" if all_same_dims else "Dynamic"
    alphas = f"{alphas_list[0]}" if all_same_alphas else "Dynamic"

    return merged_sd


def merge_to_sd_model(text_encoder, unet, models, ratios, merge_dtype='cuda'):
    text_encoder.to(merge_dtype)
    unet.to(merge_dtype)

    # create module map
    name_to_module = {}
    for i, root_module in enumerate([text_encoder, unet]):
        if i == 0:
            prefix = 'lora_te'
            target_replace_modules = ['CLIPAttention', 'CLIPMLP']
        else:
            prefix = 'lora_unet'
            target_replace_modules = (
                ['Transformer2DModel'] + ['ResnetBlock2D', 'Downsample2D', 'Upsample2D']
            )

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "Conv2d":
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        name_to_module[lora_name] = child_module

    for model, ratio in zip(models, ratios):
        print(f"loading: {model}")
        lora_sd, _ = load_state_dict(model, merge_dtype)

        print(f"merging...")
        for key in lora_sd.keys():
            if "lora_down" in key:
                up_key = key.replace("lora_down", "lora_up")
                alpha_key = key[: key.index("lora_down")] + "alpha"

                # find original module for this layer
                module_name = ".".join(key.split(".")[:-2])  # remove trailing ".lora_down.weight"
                if module_name not in name_to_module:
                    print(f"no module found for weight: {key}")
                    continue
                module = name_to_module[module_name]
                # print(f"apply {key} to {module}")

                down_weight = lora_sd[key]
                up_weight = lora_sd[up_key]

                dim = down_weight.size()[0]
                alpha = lora_sd.get(alpha_key, dim)
                scale = alpha / dim

                # W <- W + U * D
                weight = module.weight
                if len(weight.size()) == 2:
                    # linear
                    if len(up_weight.size()) == 4:  # use linear projection mismatch
                        up_weight = up_weight.squeeze(3).squeeze(2)
                        down_weight = down_weight.squeeze(3).squeeze(2)
                    weight = weight + ratio * (up_weight @ down_weight) * scale
                elif down_weight.size()[2:4] == (1, 1):
                    # conv2d 1x1
                    weight = (
                        weight
                        + ratio
                        * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                        * scale
                    )
                else:
                    # conv2d 3x3
                    conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                    # print(conved.size(), weight.size(), module.stride, module.padding)
                    weight = weight + ratio * conved * scale

                module.weight = torch.nn.Parameter(weight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sd_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--loras', type=str, nargs='+')
    parser.add_argument('--ratios', type=float, nargs='+')
    parser.add_argument('--output_path', type=str, default=None)

    args = parser.parse_args()

    pipe = DiffusionPipeline.from_pretrained(
        args.sd_path,
        custom_pipeline="lpw_stable_diffusion",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipe = pipe.to('cuda')

    merge_to_sd_model(pipe.text_encoder, pipe.unet, args.loras, args.ratios, 'cuda')
