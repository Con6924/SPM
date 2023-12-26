import json
import os
import random
from argparse import ArgumentParser
from prettytable import PrettyTable
from tqdm import tqdm

from src.configs.generation_config import GenerationConfig

from ..misc.clip_templates import anchor_templates, imagenet_templates
from .eval_util import clip_eval_by_image
from .evaluator import Evaluator, GenerationDataset


class ClipTemplateDataset(GenerationDataset):
    def __init__(
        self,
        concepts: list[str],
        save_folder: str = "benchmark/generated_imgs/",
        base_cfg: GenerationConfig = GenerationConfig(),
        num_templates: int = 80,
        num_images_per_template: int = 10,
        **kwargs
    ):
        assert 1 <= num_templates <= 80, "num_templates should be in range(1, 81)."
        meta = {}
        self.data = []
        for concept in concepts:
            meta[concept] = {}
            sampled_template_indices = random.sample(range(80), num_templates)
            for template_idx in sampled_template_indices:
                # construct cfg
                cfg = base_cfg.copy()
                cfg.prompts = [imagenet_templates[template_idx].format(concept)]
                cfg.generate_num = num_images_per_template
                cfg.save_path = os.path.join(
                    save_folder,
                    concept,
                    f"{template_idx}" + "_{}.png",
                )
                self.data.append(cfg.dict())
                # construct meta
                meta[concept][template_idx] = [
                    cfg.save_path.format(i) for i in range(num_images_per_template)
                ]
        os.makedirs(save_folder, exist_ok=True)
        meta_path = os.path.join(save_folder, "meta.json")
        print(f"Saving metadata to {meta_path} ...")
        with open(meta_path, "w") as f:
            json.dump(meta, f)


class ClipEvaluator(Evaluator):
    """
    Evaluation for CLIP-protocol accepts `save_folder` as a *JSON file* with the following format:
    {
        CONCEPT_1: {
            TEMPLATE_IDX_1_1: [IMAGE_PATH_1_1_1, IMAGE_PATH_1_1_2, ...],
            TEMPLATE_IDX_1_2: [IMAGE_PATH_1_2_1, IMAGE_PATH_1_2_2, ...],
            ...
        },
        CONCEPT_2: {
            TEMPLATE_IDX_2_1: [IMAGE_PATH_2_1_1, IMAGE_PATH_2_1_2, ...],
            TEMPLATE_IDX_2_2: [IMAGE_PATH_2_2_1, IMAGE_PATH_2_2_2, ...],
            ...
        },
        ...
    }
    CONCEPT_i: str, the i-th concept to be evaluated.
    TEMPLATE_IDX_i_j: int, range(80), the j-th selected template for CONCEPT_i.
    IMAGE_PATH_i_j_k: str, the k-th image path for CONCEPT_i, TEMPLATE_IDX_i_j.
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
        all_scores = {}
        all_cers = {}
        for concept, data in self.img_metadata.items():
            print(f"Evaluating concept:", concept)
            scores = accs = 0.0
            num_all_images = 0
            for template_idx, image_paths in tqdm(data.items()):
                template_idx = int(template_idx)
                target_prompt = imagenet_templates[template_idx].format(concept) if self.eval_with_template else concept
                anchor_prompt = anchor_templates[template_idx] if self.eval_with_template else ""
                num_images = len(image_paths)
                score, acc = clip_eval_by_image(
                    image_paths,
                    [target_prompt] * num_images,
                    [anchor_prompt] * num_images,
                )
                scores += score * num_images
                accs += acc * num_images
                num_all_images += num_images
            scores /= num_all_images
            accs /= num_all_images
            all_scores[concept] = scores
            all_cers[concept] = 1 - accs

        table = PrettyTable()
        table.field_names = ["Concept", "CLIPScore", "CLIPErrorRate"]
        for concept, score in all_scores.items():
            table.add_row([concept, score, all_cers[concept]])
        print(table)

        save_name = "evaluation_results.json" if self.eval_with_template else "evaluation_results(concept only).json"
        with open(os.path.join(self.output_path, save_name), "w") as f:
            json.dump([all_scores, all_cers], f)


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

    evaluator = ClipEvaluator(
        save_folder=args.save_folder, output_path=args.output_path
    )
    evaluator.evaluation()
