import torch_fidelity
import argparse
import os
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--original", type=str, required=True)
parser.add_argument("--generated", type=str, required=True)

args = parser.parse_args()

concepts = [f for f in os.listdir(args.original) if not (f.startswith('.') or f.startswith("coco30k")) and os.path.isdir(os.path.join(args.original, f))]

# pandas dataframe
df = pd.DataFrame(columns=['concept', 'frechet_inception_distance'])

# concept-wise metrics
for concept in concepts:
    print(f"Concept: {concept}")
    metrics = torch_fidelity.calculate_metrics(
        input1=os.path.join(args.generated, concept),
        input2=os.path.join(args.original, concept),
        cuda=True,
        fid=True,
        samples_find_deep=True)
    df = df.append({'concept': concept, **metrics}, ignore_index=True)

model_name = args.generated.split('/')[-1]
save_dir = f"output/evaluation_results/{model_name}"
os.makedirs(save_dir, exist_ok=True)
df.to_csv(f"output/evaluation_results/{model_name}/metrics.csv", index=False)
