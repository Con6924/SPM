import json
import os
from argparse import ArgumentParser

import pandas as pd
from prettytable import PrettyTable

from src.configs.generation_config import GenerationConfig

from .eval_util import clip_score
from .evaluator import Evaluator, GenerationDataset

ARTWORK_DATASETS = {
    "art": "benchmark/art_prompts.csv",
    "artwork": "benchmark/artwork_prompts.csv",
    "big_artists": "benchmark/big_artist_prompts.csv",
    "famous_art": "benchmark/famous_art_prompts.csv",
    "generic_artists": "benchmark/generic_artists_prompts.csv",
    "kelly": "benchmark/kelly_prompts.csv",
    "niche_art": "benchmark/niche_art_prompts.csv",
    "short_niche_art": "benchmark/short_niche_art_prompts.csv",
    "short_vangogh": "benchmark/short_vangogh_prompts.csv",
    "vangogh": "benchmark/vangogh_prompts.csv",
    "picasso": "benchmark/picasso_prompts.csv",
    "rembrandt": "benchmark/rembrandt_prompts.csv",
    "andy_warhol": "benchmark/andy_warhol_prompts.csv",
    "caravaggio": "benchmark/caravaggio_prompts.csv",
}


class ArtworkDataset(GenerationDataset):
    def __init__(
        self,
        datasets: list[str],
        save_folder: str = "benchmark/generated_imgs/",
        base_cfg: GenerationConfig = GenerationConfig(),
        num_images_per_prompt: int = 20,
        **kwargs,
    ) -> None:
        assert all([dataset in ARTWORK_DATASETS for dataset in datasets]), (
            f"datasets should be a subset of {ARTWORK_DATASETS}, " f"got {datasets}."
        )

        meta = {}
        self.data = []
        for dataset in datasets:
            meta[dataset] = {}
            df = pd.read_csv(ARTWORK_DATASETS[dataset])
            for idx, row in df.iterrows():
                cfg = base_cfg.copy()
                cfg.prompts = [row["prompt"]]
                cfg.seed = row["evaluation_seed"]
                cfg.generate_num = num_images_per_prompt
                cfg.save_path = os.path.join(
                    save_folder,
                    dataset,
                    f"{idx}" + "_{}.png",
                )
                self.data.append(cfg.dict())
                meta[dataset][row["prompt"]] = [
                    cfg.save_path.format(i) for i in range(num_images_per_prompt)
                ]
        os.makedirs(save_folder, exist_ok=True)
        meta_path = os.path.join(save_folder, "meta.json")
        print(f"Saving metadata to {meta_path} ...")
        with open(meta_path, "w") as f:
            json.dump(meta, f)


class ArtworkEvaluator(Evaluator):
    """
    Evaluation for artwork on CLIP-protocol accepts `save_folder` as a *JSON file* with the following format:
    {
        DATASET_1: {
            PROMPT_1_1: [IMAGE_PATH_1_1_1, IMAGE_PATH_1_1_2, ...],
            PROMPT_1_2: [IMAGE_PATH_1_2_1, IMAGE_PATH_1_2_2, ...],
            ...
        },
        DATASET_2: {
            PROMPT_2_1: [IMAGE_PATH_2_1_1, IMAGE_PATH_2_1_2, ...],
            PROMPT_2_2: [IMAGE_PATH_2_2_1, IMAGE_PATH_2_2_2, ...],
            ...
        },
        ...
    }
    DATASET_i: str, the i-th concept to be evaluated.
    PROMPT_i_j: int, the j-th prompt in DATASET_i.
    IMAGE_PATH_i_j_k: str, the k-th image path for DATASET_i, PROMPT_i_j.
    """

    def __init__(
        self,
        save_folder: str = "benchmark/generated_imgs/",
        output_path: str = "benchmark/results/",
        eval_with_template: bool = False,
    ):
        super().__init__(save_folder=save_folder, output_path=output_path)
        self.img_metadata = json.load(open(os.path.join(self.save_folder, "meta.json")))
        self.eval_with_template = eval_with_template

    def evaluation(self):
        scores = {}
        for dataset, data in self.img_metadata.items():
            score = 0.0
            num_images = 0
            for prompt, img_paths in data.items():
                score += clip_score(
                    img_paths,
                    [prompt] * len(img_paths) if self.eval_with_template else [dataset.replace("_", " ")] * len(img_paths),
                ).mean().item() * len(img_paths)
                num_images += len(img_paths)
            scores[dataset] = score / num_images

        table = PrettyTable()
        table.field_names = ["Dataset", "CLIPScore"]
        for dataset, score in scores.items():
            table.add_row([dataset, score])
        print(table)

        with open(os.path.join(self.output_path, "scores.json"), "w") as f:
            json.dump(scores, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--save_folder",
        type=str,
        help="path to json that contains metadata for generated images.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="path to save evaluation results.",
    )
    args = parser.parse_args()

    evaluator = ArtworkEvaluator(
        save_folder=args.save_folder, output_path=args.output_path
    )
    evaluator.evaluation()
