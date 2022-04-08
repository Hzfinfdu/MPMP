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
parser.add_argument("--n_prompt_tokens", default=50, type=int)
parser.add_argument("--intrinsic_dim", default=500, type=int)
parser.add_argument("--save_every", default=10000, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--n_steps", default=400000, type=int)
parser.add_argument("--print_every", default=1000, type=int)
parser.add_argument("--eval_every", default=50000, type=int)
parser.add_argument("--device", default='cuda', type=str)
parser.add_argument("--n_prompts", default=4, type=int)
parser.add_argument("--random_proj", default='he', type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--lr_router", default=.005, type=float)
parser.add_argument("--lr_prompt", default=.001, type=float)
parser.add_argument("--anneal_rate", default=8e-9, type=float)
parser.add_argument("--anneal_min", default=.05, type=float)
args = parser.parse_args()

class Optim:
    def __init__(self, para1, para2, lr1, lr2):
        self.optimizer1 = AdamW(para1, lr=lr1, betas=(0.9, 0.999))
        self.optimizer2 = AdamW(para2, lr=lr2, betas=(0.9, 0.999))

    def step(self):
        self.optimizer1.step()
        self.optimizer2.step()

    def zero_grad(self):
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

save_path = f'./results/PromptTokens{args.n_prompt_tokens}_IntrinsicDim{args.intrinsic_dim}_BatchSize{args.batch_size}_NPrompts{args.n_prompts}_LrRouter{args.lr_router}_LrPrompt{args.lr_prompt}'
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
args.save_path = save_path
torch.manual_seed(args.seed)

model = PretrainPrompt(args.intrinsic_dim, args.n_prompt_tokens, num_datasets, args.n_prompts)
model.prompt_embed_model.load_state_dict(torch.load('/remote-home/zfhe/projects/BBT-prompt_pretrain/results/PromptTokens50_IntrinsicDim500_BatchSize8_NPrompts4_LrRouter0.005_LrPrompt0.001/models/399999.th'))
optimizer = Optim([model.prompt_embed_model.prompt_logits], [model.prompt_embed_model.AZ], args.lr_router, args.lr_prompt)
# optimizer = AdamW(model.prompt_embed_model.parameters(), lr=args.lr, betas=(0.9, 0.999))
trainer = MutitaskTrainer(args, model, optimizer)
trainer.train()