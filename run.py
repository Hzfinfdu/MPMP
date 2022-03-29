import torch
from model import PretrainPrompt
from trainer import MutitaskTrainer
from dataload import *
import torch
import argparse
from torch.optim import AdamW
import os


parser = argparse.ArgumentParser()
# parser.add_argument("--task_name", default='sst2', type=str)
parser.add_argument("--n_prompt_tokens", default=30, type=int)
parser.add_argument("--intrinsic_dim", default=500, type=int)
parser.add_argument("--save_every", default=1000, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--n_steps", default=200000, type=int)
parser.add_argument("--print_every", default=50, type=int)
parser.add_argument("--eval_every", default=3000, type=int)
parser.add_argument("--device", default='cuda', type=str)
parser.add_argument("--n_prompts", default=3, type=int)
parser.add_argument("--random_proj", default='he', type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--anneal_every", default=2000, type=int)
args = parser.parse_args()


save_path = f'./results/PromptTokens{args.n_prompt_tokens}_IntrinsicDim{args.intrinsic_dim}_BatchSize{args.batch_size}_NPrompts{args.n_prompts}'
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
args.save_path = save_path
torch.manual_seed(args.seed)


model = PretrainPrompt(args.intrinsic_dim, args.n_prompt_tokens, num_datasets, args.n_prompts)
optimizer = AdamW(model.prompt_embed_model.parameters(), lr=args.lr, betas=(0.9, 0.999))
# scheduler =
trainer = MutitaskTrainer(args, model, optimizer)
trainer.train()