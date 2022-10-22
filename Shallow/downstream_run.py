import torch
from model import PretrainPrompt
from trainer import MutitaskTrainer, DownstreamTrainer
from dataloader import *
import torch
import argparse
from torch.optim import AdamW
import os


parser = argparse.ArgumentParser()
parser.add_argument("--n_prompt_tokens", default=50, type=int)
parser.add_argument("--intrinsic_dim", default=300, type=int)
parser.add_argument("--save_every", default=10000, type=int)
parser.add_argument("--batch_size", default=32, type=int)
# parser.add_argument("--n_steps", default=2000000, type=int)
parser.add_argument("--n_epochs", default=1000, type=int)
parser.add_argument("--print_every", default=25, type=int)
parser.add_argument("--eval_every", default=25, type=int)
parser.add_argument("--device", default='cuda:0', type=str)
parser.add_argument("--n_prompts", default=8, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--data_num", default=32, type=int)
parser.add_argument("--lr_router", default=3e-3, type=float)
parser.add_argument("--lr_prompt", default=1e-30, type=float)
parser.add_argument("--anneal_rate", default=None, type=float)
parser.add_argument("--anneal_min", default=None, type=float)
parser.add_argument("--init_temperature", default=1., type=float)
parser.add_argument("--step_size1", default=500, type=int)
parser.add_argument("--step_size2", default=500, type=int)
parser.add_argument("--gamma1", default=1e-30, type=float)
parser.add_argument("--gamma2", default=3e26, type=float)
parser.add_argument("--is_downstream", default=True, type=bool)
parser.add_argument("--task_name", default='lcqmc', type=str)
args = parser.parse_args()

label_dict = {
    'chnsenticorp': 2,
    'lcqmc': 2,
    'tnews': 14,
    'amazon': 5,
    'drcd': 1,
    'cmnli': 3,
    'thucnews': 10,
    'bq': 2,
    'cmrc2018': 1,
    'ccpm': 1,
    'cotemfw': 1,
    'chnsent': 2,
    'ocnli': 3,
    'c3': 1,
    'cotebd': 1,
}

num_labels = label_dict[args.task_name]
args.k_shot = args.data_num // num_labels if num_labels < 10 else 8

# args.batch_size = args.k_shot * num_labels
# args.step_size1 = args.step_size2 = (args.k_shot * num_labels / args.batch_size * args.n_epochs // 2)


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

save_path = f'/remote-home/zfhe/projects/MPMP/BBTv1/downstream_results'
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
args.save_path = save_path
torch.manual_seed(args.seed)

model = PretrainPrompt(args.intrinsic_dim, args.n_prompt_tokens, 1, args.n_prompts, args.init_temperature)

state = torch.load('/home/ma-user/work/zfhe/best_v1.th')
# model.model.model.encoder.encoder.A = state['A']
# model.model.model.encoder.encoder.z = state['z']
# model.model.model.encoder.encoder.prompt = torch.nn.Parameter(data=torch.mm(state['z'], state['A']))
task_num = -1
if task_num >= 0:  # initialize with a seen task
    model.model.model.encoder.encoder.router.data = state['router'][task_num].unsqueeze(0)
# model.model.model.encoder.encoder.router = state['router']
# model.model.qa_outputs.weight = state['lmhead']
optimizer = Optim(
    [model.model.model.encoder.encoder.router],
    [
        model.model.model.encoder.encoder.prompt,
        # model.model.model.encoder.encoder.A,
        # model.model.model.encoder.encoder.z,
        model.model.qa_outputs.weight
    ],
    args.lr_router,
    args.lr_prompt
)
args.step_size1 = None
if args.step_size1 is not None and args.step_size2 is not None and args.gamma1 is not None and args.gamma2 is not None:
    scheduler = Scheduler(optimizer, args.step_size1, args.step_size2, args.gamma1, args.gamma2)
else:
    scheduler = None
args.save_path = f'/remote-home/zfhe/projects/MPMP/BBTv1/downstream_results'
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path, exist_ok=True)
trainer = DownstreamTrainer(args, model, optimizer, scheduler)
test_acc = trainer.train()
with open('res.txt', 'a+') as f:
    print(f'task name {args.task_name}, seed {args.seed}, acc {test_acc}', file=f)