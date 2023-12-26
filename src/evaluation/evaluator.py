import os
from typing import Iterator

from torch.utils.data import IterableDataset



class GenerationDataset(IterableDataset):
    """
    Dataset for generate images.
    """

    def __init__(self) -> None:
        super().__init__(self)
        self.data = []

    def __iter__(self) -> Iterator[dict]:
        return iter(self.data)


class Evaluator:
    """
    Base evaluator for generated images.

    Args:
        cfg (GenerationConfig): config for image generation.
        save_images (bool): whether to save generated images for evaluation.
        save_folder (str): path to save generated images, ignored if save_images is False.
            *Recommend*: f"benchmark/generated_imgs/{model_name}/"
        output_path (str): path to save evaluation results.
            *Recommend*: f"benchmark/results/{model_name}/"
    """

    def __init__(
        self,
        save_folder: str = "benchmark/generated_imgs/",
        output_path: str = "benchmark/results/",
    ):
        self.save_folder = save_folder
        self.output_path = output_path

        if not os.path.exists(self.save_folder):
            raise FileNotFoundError(f"Image path {self.save_folder} not found.")

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def evaluation(self):
        pass
