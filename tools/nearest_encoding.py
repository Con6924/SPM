import os
import torch
import argparse
from tqdm import tqdm
from prettytable import PrettyTable

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir) 

from src.engine import train_util as train_util
import src.models.model_util as model_util
from src.configs import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding_model", choices=["sd14", "sd20"], default="sd14")
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument("--num_neighbors", type=int, default=20)
    parser.add_argument("--precision", choices=['float16', 'float32'], default='float32')

    args = parser.parse_args()

    if args.encoding_model == "sd14":
        dir_ = "CompVis/stable-diffusion-v1-4"
    else:
        raise NotImplementedError
    weight_dtype = config.parse_precision(args.precision)
    
    tokenizer, text_encoder, unet, pipe = model_util.load_checkpoint_model(
        dir_, 
        v2=args.encoding_model=="sd20", 
        weight_dtype=weight_dtype
    )

    vocab = list(tokenizer.decoder.values())
    if os.path.exists(f"src/misc/{args.encoding_model}-token-encodings.pt"):
        all_encodings = torch.load(f"src/misc/{args.encoding_model}-token-encodings.pt")
    else:
        print(f"Generating token encodings from {dir_} ...")
        all_encodings = []
        for i, word in tqdm(enumerate(vocab)):
            token = train_util.text_tokenize(tokenizer, word)
            all_encodings.append(text_encoder(token.to(text_encoder.device))[0].detach().cpu())
            if i % 100 == 0:
                torch.cuda.empty_cache()
        torch.save(all_encodings, 'output/generated_images/sd14-token-encodings.pt')
        torch.cuda.empty_cache()


    all_encodings = torch.concatenate(all_encodings)

    inp_token = train_util.text_tokenize(tokenizer, args.concept)
    inp_encodings = text_encoder(inp_token.to(text_encoder.device))[0].detach().cpu()

    scores = torch.cosine_similarity(all_encodings.flatten(1,-1), inp_encodings.flatten(1,-1).cpu(), dim=-1)
    sorted_scores, sorted_ids = torch.sort(scores, descending=True)
    
    table = PrettyTable()
    table.field_names = ["Concept", "Similarity", "TokenID"]
    for emb_score, emb_id in zip(sorted_scores[0: args.num_neighbors], \
                                 sorted_ids[0: args.num_neighbors]):
        emb_name = vocab[emb_id.item()]

        table.add_row([emb_name, emb_score.item()*100, emb_id.item()])

    table.float_format=".3"
    print(table)