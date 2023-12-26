import random
import torch

from src.configs.prompt import PromptEmbedsPair


def sample(prompt_pair: PromptEmbedsPair, tokenizer=None, text_encoder=None):
    samples = []
    while len(samples) < prompt_pair.sampling_batch_size:
        while True:
            # sample from gaussian distribution
            noise = torch.randn_like(prompt_pair.target)
            # normalize the noise
            noise = noise / noise.view(-1).norm(dim=-1)
            # compute the similarity
            sim = torch.cosine_similarity(prompt_pair.target.view(-1), noise.view(-1), dim=-1)
            # the possibility of accepting the sample = 1 - sim
            if random.random() < 1 - sim:
                break
        scale = random.random() * 0.4 + 0.8
        sample = scale * noise * prompt_pair.target.view(-1).norm(dim=-1)
        samples.append(sample)
    
    samples = [torch.cat([prompt_pair.unconditional, s]) for s in samples]
    samples = torch.cat(samples, dim=0)
    return samples
    
def sample_xl(prompt_pair: PromptEmbedsPair, tokenizers=None, text_encoders=None):
    res = []
    for unconditional, target in zip(
        [prompt_pair.unconditional.text_embeds, prompt_pair.unconditional.pooled_embeds],
        [prompt_pair.target.text_embeds, prompt_pair.target.pooled_embeds]
    ):
        samples = []
        while len(samples) < prompt_pair.sampling_batch_size:
            while True:
                # sample from gaussian distribution
                noise = torch.randn_like(target)
                # normalize the noise
                noise = noise / noise.view(-1).norm(dim=-1)
                # compute the similarity
                sim = torch.cosine_similarity(target.view(-1), noise.view(-1), dim=-1)
                # the possibility of accepting the sample = 1 - sim
                if random.random() < 1 - sim:
                    break
            scale = random.random() * 0.4 + 0.8
            sample = scale * noise * target.view(-1).norm(dim=-1)
            samples.append(sample)
        
        samples = [torch.cat([unconditional, s]) for s in samples]
        samples = torch.cat(samples, dim=0)
        res.append(samples)
    
    return res
