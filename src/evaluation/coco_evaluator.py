import json
import os
from argparse import ArgumentParser

import pandas as pd
from prettytable import PrettyTable
from cleanfid import fid

from src.configs.generation_config import GenerationConfig

from .evaluator import Evaluator, GenerationDataset


class Coco30kGenerationDataset(GenerationDataset):
    """
    Dataset for COCO-30k Caption dataset.
    """
    def __init__(
        self,
        save_folder: str = "benchmark/generated_imgs/",
        base_cfg: GenerationConfig = GenerationConfig(),
        data_path: str = "benchmark/coco_30k.csv",
        **kwargs
    ) -> None:
        df = pd.read_csv(data_path)
        self.data = []
        for idx, row in df.iterrows():
            cfg = base_cfg.copy()
            cfg.prompts = [row["prompt"]]
            cfg.negative_prompt = ""
            # fix width & height to be divisible by 8
            cfg.width = row["width"] - row["width"] % 8
            cfg.height = row["height"] - row["height"] % 8
            cfg.seed = row["evaluation_seed"]
            cfg.generate_num = 1
            cfg.save_path = os.path.join(
                save_folder,
                "coco30k",
                "COCO_val2014_" + "%012d" % row["image_id"] + ".jpg",
            )
            self.data.append(cfg.dict())


class CocoEvaluator(Evaluator):
    """
    Evaluator on COCO-30k Caption dataset.
    """
    def __init__(self, 
                 save_folder: str = "benchmark/generated_imgs/", 
                 output_path: str = "benchmark/results/",
                 data_path: str = "/jindofs_temp/users/406765/COCOCaption/30k",
    ):
        super().__init__(save_folder=save_folder, output_path=output_path)

        self.data_path = data_path

    def evaluation(self):
        print("Evaluating on COCO-30k Caption dataset...")
        fid_value = fid.compute_fid(os.path.join(self.save_folder, "coco30k"), self.data_path)
        # metrics = torch_fidelity.calculate_metrics(
        #     input1=os.path.join(self.save_folder, "coco30k"),
        #     input2=self.data_path,
        #     cuda=True,
        #     fid=True,
        #     samples_find_deep=True)

        pt = PrettyTable()
        pt.field_names = ["Metric", "Value"]
        pt.add_row(["FID", fid_value])
        print(pt)
        with open(os.path.join(self.output_path, "coco-fid.json"), "w") as f:
            json.dump({"FID": fid_value}, f)
