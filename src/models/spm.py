# ref:
# - https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# - https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

import os
import math
from typing import Optional, List

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file


class SPMLayer(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        spm_name,
        org_module: nn.Module,
        multiplier=1.0,
        dim=4,
        alpha=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.spm_name = spm_name
        self.dim = dim

        if org_module.__class__.__name__ == "Linear":
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, dim, bias=False)
            self.lora_up = nn.Linear(dim, out_dim, bias=False)

        elif org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.dim = min(self.dim, in_dim, out_dim)
            if self.dim != dim:
                print(f"{spm_name} dim (rank) is changed to: {self.dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(self.dim, out_dim, (1, 1), (1, 1), bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # same as microsoft's
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        return (
            self.org_forward(x)
            + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        )
        

class SPMNetwork(nn.Module):
    UNET_TARGET_REPLACE_MODULE_TRANSFORMER = [
        "Transformer2DModel",
    ]
    UNET_TARGET_REPLACE_MODULE_CONV = [
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D",
    ]

    SPM_PREFIX_UNET = "lora_unet"   # aligning with SD webui usage
    DEFAULT_TARGET_REPLACE = UNET_TARGET_REPLACE_MODULE_TRANSFORMER

    def __init__(
        self,
        unet: UNet2DConditionModel,
        rank: int = 4,
        multiplier: float = 1.0,
        alpha: float = 1.0,
        module = SPMLayer,
        module_kwargs = None,
    ) -> None:
        super().__init__()

        self.multiplier = multiplier
        self.dim = rank
        self.alpha = alpha

        self.module = module
        self.module_kwargs = module_kwargs or {}

        # unet spm
        self.unet_spm_layers = self.create_modules(
            SPMNetwork.SPM_PREFIX_UNET,
            unet,
            SPMNetwork.DEFAULT_TARGET_REPLACE,
            self.dim,
            self.multiplier,
        )
        print(f"Create SPM for U-Net: {len(self.unet_spm_layers)} modules.")

        spm_names = set()
        for spm_layer in self.unet_spm_layers:
            assert (
                spm_layer.spm_name not in spm_names
            ), f"duplicated SPM layer name: {spm_layer.spm_name}. {spm_names}"
            spm_names.add(spm_layer.spm_name)

        for spm_layer in self.unet_spm_layers:
            spm_layer.apply_to()
            self.add_module(
                spm_layer.spm_name,
                spm_layer,
            )

        del unet

        torch.cuda.empty_cache()

    def create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_modules: List[str],
        rank: int,
        multiplier: float,
    ) -> list:
        spm_layers = []

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ["Linear", "Conv2d"]:
                        spm_name = prefix + "." + name + "." + child_name
                        spm_name = spm_name.replace(".", "_")
                        print(f"{spm_name}")
                        spm_layer = self.module(
                            spm_name, child_module, multiplier, rank, self.alpha, **self.module_kwargs
                        )
                        spm_layers.append(spm_layer)

        return spm_layers

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        all_params = []

        if self.unet_spm_layers:
            params = []
            [params.extend(spm_layer.parameters()) for spm_layer in self.unet_spm_layers]
            param_data = {"params": params}
            if default_lr is not None:
                param_data["lr"] = default_lr
            all_params.append(param_data)

        return all_params

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        for key in list(state_dict.keys()):
            if not key.startswith("lora"):
                del state_dict[key]

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def __enter__(self):
        for spm_layer in self.unet_spm_layers:
            spm_layer.multiplier = 1.0

    def __exit__(self, exc_type, exc_value, tb):
        for spm_layer in self.unet_spm_layers:
            spm_layer.multiplier = 0
