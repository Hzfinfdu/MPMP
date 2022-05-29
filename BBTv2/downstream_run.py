import torch
from model import PretrainPrompt
from trainer import MutitaskTrainer, DownstreamTrainer
from dataload import *
import torch
import argparse
from torch.optim import AdamW
import os


parser = argparse.ArgumentParser()
parser.add_argument("--n_prompt_tokens", default=50, type=int)
parser.add_argument("--intrinsic_dim", default=300, type=int)
parser.add_argument("--save_every", default=10000, type=int)
parser.add_argument("--batch_size", default=7, type=int)
# parser.add_argument("--n_steps", default=2000000, type=int)
parser.add_argument("--n_epochs", default=1000, type=int)
parser.add_argument("--print_every", default=25, type=int)
parser.add_argument("--eval_every", default=25, type=int)
parser.add_argument("--device", default='cuda:0', type=str)
parser.add_argument("--n_prompts", default=8, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--lr_router", default=1e-3, type=float)
parser.add_argument("--lr_prompt", default=1e-11, type=float)
parser.add_argument("--anneal_rate", default=None, type=float)
parser.add_argument("--anneal_min", default=None, type=float)
parser.add_argument("--init_temperature", default=1., type=float)
parser.add_argument("--step_size1", default=500 * 5, type=int)
parser.add_argument("--step_size2", default=500 * 5, type=int)
parser.add_argument("--gamma1", default=1e-15, type=float)
parser.add_argument("--gamma2", default=2e6, type=float)
parser.add_argument("--is_downstream", default=True, type=bool)
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

    def state_dict(self):
        return {
            'optimizer1': self.optimizer1.state_dict(),
            'optimizer2': self.optimizer2.state_dict()
        }


class Scheduler:
    def __init__(self, optim, step1=10000, step2=10000, gamma1=.1, gamma2=.1):
        self.scheduler1 = torch.optim.lr_scheduler.StepLR(optim.optimizer1, step1, gamma1)
        self.scheduler2 = torch.optim.lr_scheduler.StepLR(optim.optimizer2, step2, gamma2)

    def step(self):
        self.scheduler1.step()
        self.scheduler2.step()


torch.manual_seed(args.seed)

model = PretrainPrompt(args.intrinsic_dim, args.n_prompt_tokens, 1, args.n_prompts, args.init_temperature)
# model.prompt_embed_model.load_state_dict(torch.load('/remote-home/zfhe/projects/BBT-prompt_pretrain/results/PromptTokens50_IntrinsicDim500_BatchSize8_NPrompts4_LrRouter0.005_LrPrompt0.001/models/399999.th'))
state = torch.load('/home/ma-user/work/zfhe/BBTPrefixPretraining/results/PromptTokens50_BatchSize32_NPrompts8_LrRouter0.0005_LrPrompt0.0001_AnnealParams1.0;None;None/best.th')
model.model.model.encoder.encoder.A = state['A']
model.model.model.encoder.encoder.z = state['z']
task_num = -1
if not task_num < 0:
    model.model.model.encoder.encoder.router.data = state['router'][task_num].unsqueeze(0)
# model.model.model.encoder.encoder.router = state['router']
model.model.qa_outputs.weight = state['lmhead']
optimizer = Optim(
    [model.model.model.encoder.encoder.router],
    [
        model.model.model.encoder.encoder.A,
        model.model.model.encoder.encoder.z,
        model.model.qa_outputs.weight
    ],
    args.lr_router,
    args.lr_prompt
)
if args.step_size1 is not None and args.step_size2 is not None and args.gamma1 is not None and args.gamma2 is not None:
    scheduler = Scheduler(optimizer, args.step_size1, args.step_size2, args.gamma1, args.gamma2)
else:
    scheduler = None
args.save_path = f'/home/ma-user/work/zfhe/BBTPrefixPretraining/downstream_results/PromptTokens50_BatchSize32_NPrompts8_LrRouter0.005_LrPrompt0.001_AnnealParams1.0;None;None_downstream'
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path, exist_ok=True)
trainer = DownstreamTrainer(args, model, optimizer, scheduler)
trainer.train()