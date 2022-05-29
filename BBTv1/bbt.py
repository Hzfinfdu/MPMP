import os
import copy
import time
import math
import random

import torch
# import fitlog
import argparse
import numpy as np
import cma
from fastNLP import cache_results, Tester, DataSet
from transformers import BertTokenizer, BartConfig as CPTConfig
from model import PretrainPrompt
from sklearn.metrics import f1_score
from dataload import *
from bayes_opt import BayesianOptimization
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default='chnsenticorp', type=str)
parser.add_argument("--k_shot", default=16, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--budget_router", default=200, type=int)
parser.add_argument("--budget_prompt", default=4000, type=int)
parser.add_argument("--popsize", default=20, type=int)
parser.add_argument("--bound", default=5, type=int)
parser.add_argument("--print_every", default=50, type=int)
parser.add_argument("--eval_every", default=100, type=int)
parser.add_argument("--device", default='cuda:0', type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--init_prompt_path", default=None, type=str)
parser.add_argument("--cat_or_add", default='add', type=str)
parser.add_argument("--parallel", action='store_true', help='Whether to allow parallel evaluation')
parser.add_argument(
    "--inference_framework",
    default='pt',
    type=str,
    help='''Which inference framework to use. 
         Currently supports `pt` and `ort`, standing for pytorch and Microsoft onnxruntime respectively'''
)
parser.add_argument(
    "--onnx_model_path",
    default=None,
    type=str,
    help='Path to your onnx model.'
)
args = parser.parse_args()

task_name = args.task_name
intrinsic_dim = 300  # fixed according to pretrain config
k_shot = args.k_shot
batch_size = args.batch_size
budget_router = args.budget_router
budget_prompt = args.budget_prompt
bound = args.bound
# bound = math.sqrt(intrinsic_dim)
# if random_proj == 'normal':
#     bound = math.pow(intrinsic_dim, 0.75)
# elif model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
#     bound = 100
# else:
#     bound = 5
if args.popsize > 0:
    popsize = args.popsize
else:
    popsize = 4 + 3 * np.log(intrinsic_dim)
device = args.device
seed = args.seed
print_every = args.print_every
eval_every = args.eval_every
# if task_name in ['mrpc', 'snli', 'qnli', 'rte']:
#     args.cat_or_add = 'cat'
cat_or_add = args.cat_or_add
parallel = args.parallel
inference_framework = args.inference_framework
onnx_model_path = args.onnx_model_path
init_prompt_path = args.init_prompt_path
assert init_prompt_path is not None, 'Pretrained path is required.'
if task_name == 'c3':
    batch_size = 1

if inference_framework not in ['pt', 'ort']:
    raise ValueError(f'inference_framework only supports "pt", "ort", got `{inference_framework}` instead.')
if inference_framework == 'ort':
    assert onnx_model_path is not None, 'Path to onnx model is required, got None instead.'
    assert os.path.exists(onnx_model_path), f'In valid onnx model path `{onnx_model_path}`'


save_path = 'bbt_results/{}_results/range_{}_budget_{}_{}_seed_{}_{}_{}_{}'.format(
    task_name,
    bound,
    budget_router,
    budget_prompt,
    seed,
    cat_or_add,
    'parallel' if parallel else 'serial',
    inference_framework
)
print('Results will be saved in {}'.format(save_path))

if os.path.exists(save_path):
    print('Experiment already run.')
    exit()

args.save_path = save_path
args.bbt_version = 'bbt'

# log_dir = './logs'
# fitlog.set_log_dir(log_dir)
# fitlog.commit(__file__, fit_msg=save_path)
# fitlog.add_hyper(args)
# fitlog.add_hyper_in_file(__file__)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class LMForwardAPI:
    def __init__(self):
        self.model = PretrainPrompt(d=300, prompt_token_num=50, n_tasks=1, n_prompts=8, init_temperature=1.)  # pretrain params, no need for tuning
        pretrained_state = torch.load(init_prompt_path)
        self.model.model.model.encoder.encoder.A.data = pretrained_state['A']
        self.model.model.model.encoder.encoder.z.data = pretrained_state['z']
        self.model.model.qa_outputs.weight = pretrained_state['lmhead']

        self.config = CPTConfig.from_pretrained('fnlp/cpt-large')
        self.init_prompt = None
        self.model.to(device)
        self.model.eval()
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_prompt = pretrained_state['z']
        self.best_router = None
        self.num_call = 0
        # self.save_path = save_path
        self.print_every = print_every
        self.eval_every = eval_every
        # if save_path is not None:
        #     os.makedirs(save_path, exist_ok=True)

    def set_target_param(self, param, is_router=False, interest_index=None):
        if is_router:
            self.model.model.model.encoder.encoder.router.data = param.unsqueeze(0)
        else:
            self.model.model.model.encoder.encoder.z.data[interest_index] = param

    def eval(self, target_param=None, test_data=None, is_router=False, interest_index=None):
        if test_data is not None:
            self.set_target_param(torch.tensor(self.best_router, device=device, dtype=torch.float32), True)
            self.model.model.model.encoder.encoder.z.data = self.best_prompt
            testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate)
            with torch.no_grad():
                total_acc, n_batchs = 0., 0
                for i, iter in tqdm(enumerate(testloader)):
                    for k, v in iter.items():
                        if iter[k] is not None:
                            iter[k] = v.to(device)
                    iter['is_train'] = False
                    _, acc = self.model(**iter)
                    total_acc += acc
                    n_batchs += 1
                total_acc /= n_batchs
            return total_acc
        else:
            self.num_call += 1

            if target_param is None:
                target_param = self.best_router if is_router else self.best_prompt
            tmp_param = copy.deepcopy(target_param)  # list or numpy.ndarray

            self.set_target_param(torch.tensor(target_param, device=device, dtype=torch.float32), is_router, interest_index)
            for k, v in train_data.items():
                if train_data[k] is not None:
                    train_data[k] = v.to(device)
            with torch.no_grad():
                loss, perf = self.model(**train_data, is_train=False)  # in order to return perf, does not affect forward
            # fitlog.add_loss(loss, name=self.loss_type, step=self.num_call)
            # fitlog.add_metric(perf, name='train_acc', step=self.num_call)

            if perf > self.best_train_perf:
                self.best_train_perf = perf
                # fitlog.add_best_metric(self.best_train_perf, name='train_acc')

            # if self.save_path is not None:
            #     with open(os.path.join(self.save_path, 'train_acc.txt'), 'a') as fout:
            #         fout.write('{}\t{}\n'.format(self.num_call, perf))

            if self.num_call % self.print_every == 0:
                print(
                    '[# API Calls {}] loss: {}. Current perf: {}. Best perf so far: {}'.format(
                        self.num_call,
                        round(float(loss), 4),
                        round(float(perf), 4),
                        round(float(self.best_train_perf), 4)))

            if self.num_call % self.eval_every == 0:
                print('********* Evaluated on dev set *********')
                for k, v in dev_data.items():
                    if dev_data[k] is not None:
                        dev_data[k] = v.to(device)
                with torch.no_grad():
                    dev_loss, dev_perf = self.model(**dev_data, is_train=False)
                # fitlog.add_metric(dev_perf, name='dev_acc', step=self.num_call)
                if dev_perf > self.best_dev_perf:
                    self.best_dev_perf = dev_perf
                    # fitlog.add_best_metric(self.best_dev_perf, name='dev_acc')
                    if is_router:
                        self.best_router = copy.deepcopy(tmp_param)
                    else:
                        self.best_prompt = self.model.model.model.encoder.encoder.z.data.clone().detach()
                # if self.save_path is not None:
                #     with open(os.path.join(self.save_path, 'dev_acc.txt'), 'a') as fout:
                #         fout.write('{}\t{}\n'.format(self.num_call, dev_loss))
                print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
                    round(float(dev_loss), 4),
                    round(float(dev_perf), 4),
                    round(float(self.best_dev_perf), 4)))
                print('********* Done *********')
            return loss.item()


tokenizer = BertTokenizer.from_pretrained('fnlp/cpt-large')

# cache_fn = f"caches/data_{model_name.replace('/', '-')}_{task_name}_{n_prompt_tokens}_{seed}.pt"
Data_config = {
    'ocnli': (OcnliDataset, 11),
    'cmnli': (CMNLIDataset, 11),
    'chnsenticorp': (ChnSentiCorpDataset, 16),
    'thucnews': (THUCNewsDataset, 8),
    'bq': (BQDataset, 16),
    'drcd': (DRCDDataset, 32),
    'c3': (C3Dataset, 32),
    'cmrc2018': (Cmrc2018Dataset, 32),
    'lcqmc': (LCQMCDataset, 16),
    'tnews': (tnewsDataset, 8),
    'cotebd': (CoteBdDataset, 32),
    'cotemfw': (CoteMfwDataset, 32),
    'ccpm': (CCPMDataset, 32),
    'amazon': (AmazonDataset, 7),
}

def collate(batch_input):
    input_ids = [torch.tensor(d['input_ids']) for d in batch_input]
    start_positions = torch.tensor([d['start_positions'] for d in batch_input])
    end_positions = torch.tensor([d['end_positions'] for d in batch_input])
    input_ids = pad_sequence(input_ids, batch_first=True)
    label_mask = None
    label = None
    if 'label_mask' in batch_input[0].keys():
        label_mask = [torch.tensor(d['label_mask']) for d in batch_input]
        label_mask = pad_sequence(label_mask, batch_first=True)
        assert label_mask.shape == input_ids.shape
        label = torch.tensor([d['label'] for d in batch_input])
    return {
        'input_ids': input_ids,
        'start_positions': start_positions,
        'end_positions': end_positions,
        'label_mask': label_mask,
        'label': label,
        'task_id': torch.tensor([0], device=device),
    }


cls, k_shot = Data_config[task_name]
data = cls().get_dataset(split='downstream', k_shot=k_shot, seed=seed)
train_data, dev_data = collate(data['train']), collate(data['dev'])
test_data = cls().get_dataset(split='test')

model_forward_api = LMForwardAPI()

start_time = time.time()
# optimize router
bounds_x = [f'x{i}' for i in range(8)]  # 8 is the number of skilled modules
bounds_range = [(-5, 5)] * 8  # 5 is fixed according to pretrained router's bound


def A_model_forward_api_eval(**kwargs):
    x = np.fromiter(kwargs.values(), dtype=float)
    return -model_forward_api.eval(x, is_router=True)


pbounds = dict(zip(bounds_x, bounds_range))
BO_optimizer = BayesianOptimization(
    f=A_model_forward_api_eval,
    pbounds=pbounds,
    verbose=0,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=seed,
)
BO_optimizer.maximize(init_points=20, n_iter=budget_router - 20, acq='ucb', kappa=2.)
best_dict_x = BO_optimizer.max['params']
best_x = np.fromiter(best_dict_x.values(), dtype=float)

model_forward_api.set_target_param(torch.tensor(model_forward_api.best_router, device=device, dtype=torch.float32), True)
print('Optimal router:', model_forward_api.best_router)
activated_prompt_index = (model_forward_api.best_router > 0.).nonzero()[0]
num_activated = len(activated_prompt_index)

# optimize prompt
cma_opts = {
    'seed': seed,
    'popsize': popsize,
    'maxiter': budget_prompt // (popsize * num_activated),
    'verbose': -1,
}
if bound > 0:
    cma_opts['bounds'] = [-1 * bound, 1 * bound]

es_list = [cma.CMAEvolutionStrategy(model_forward_api.model.model.model.encoder.encoder.z.data[idx].cpu().detach().numpy().tolist(), 0.5, inopts=cma_opts) for idx in activated_prompt_index]

for _ in range(budget_prompt // (popsize * num_activated)):
    for i, es in enumerate(es_list):
        solutions = es.ask()
        fitnesses = [model_forward_api.eval(x, interest_index=activated_prompt_index[i]) for x in solutions]
        es.tell(solutions, fitnesses)
        model_forward_api.set_target_param(torch.tensor(es.result.xbest, device=device, dtype=torch.float32), interest_index=activated_prompt_index[i])


end_time = time.time()
print('Done. Elapsed time: {} (mins)'.format((end_time - start_time) / 60))
print('Evaluate on test data...')
test_acc = model_forward_api.eval(test_data=test_data, is_router=True)
with open('res.txt', 'a+') as f:
    print(f'task {task_name}. seed {seed}. Test acc: {round(test_acc, 4)}', file=f)
# fitlog.finish()
