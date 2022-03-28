import json
import torch
import torch.nn as nn
import math
from scipy import special
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from modeling_cpt import CPTForConditionalGeneration, CPTForQuestionAnswering
from transformers import BertTokenizer


# n_tasks = 6
# n_prompts = 2
# prompt_token_num = 3
# d = 500
# D = prompt_token_num*4096
# Taks2Prompt = torch.rand(Task_num,prompt_token_num)

class PretrainPrompt(nn.Module):
    def __init__(self, d, prompt_token_num, n_tasks, n_prompts):
        super(PretrainPrompt, self).__init__()
        # self.tokenizer = BertTokenizer.from_pretrained("fnlp/cpt-large")
        self.model = CPTForQuestionAnswering.from_pretrained("fnlp/cpt-large")
        self.prompt_embed_model = PromptChoice(d, self.model.config.hidden_size, prompt_token_num, n_tasks, n_prompts)
        # self.prompt_embedding = nn.Parameter(torch.zeros(32, 50, 1024))

    def forward(self, input_ids, start_positions, end_positions, task_id,is_train = True):
        batch_size = input_ids.size()[0]
        prompt_embedding,prompt_logits,IBP_loss = self.prompt_embed_model(task_id=task_id, batch_size=batch_size,is_train=is_train)

        outputs = self.model(input_ids=input_ids, prompt_embedding=prompt_embedding, start_positions=start_positions,
                             end_positions=end_positions)
        loss = outputs.loss
        acc = torch.mul((start_positions == outputs.start_logits.argmax(dim=1)).long(),
                        (end_positions == outputs.end_logits.argmax(dim=1)).long()).sum() / batch_size
        # print("######loss=",loss)
        # print("IBPloss=",IBP_loss)
        loss = loss + IBP_loss
        return loss, acc, prompt_logits
        # loss = prompt_embedding.sum(dim=0).sum(dim=0).sum(dim=0)
        # return loss, torch.tensor(0)


class PromptChoice(nn.Module):
    def __init__(self, d, hidden_size, prompt_token_num, n_tasks, n_prompts):
        super(PromptChoice, self).__init__()
        self.prompt_logits = nn.Parameter(torch.empty((n_tasks, n_prompts)).uniform_(-1e-3, 1e-3))
        self.ones = nn.Parameter(torch.ones(n_tasks, n_prompts))
        self.zeros = nn.Parameter(torch.zeros(n_tasks, n_prompts))
        self.Z = nn.Parameter(torch.rand(n_prompts, d, 1))
        self.A = nn.Parameter(torch.rand(n_prompts, prompt_token_num * hidden_size, d))
        # self.AZ = nn.Parameter(torch.rand(n_prompts,prompt_token_num*hidden_size))
        self.hidden_size = hidden_size
        self.prompt_token_num = prompt_token_num
        self.n_tasks = n_tasks
        self.n_prompts = n_prompts
        self.EPS = 1e-12
        self.temperature = 10.

    def forward(self, task_id, batch_size,is_train):
        prompt_logits = self.prompt_logits[task_id]
        try:
            prompt_logits = RelaxedBernoulli(temperature=self.temperature, logits=prompt_logits).rsample()
        except ValueError:
            print(prompt_logits)
        prompt_logits = prompt_logits / (prompt_logits.sum(dim=-1, keepdim=True) + self.EPS)
        AZ = torch.bmm(self.A, self.Z).squeeze(-1)
        prompt_embedding = torch.mm(prompt_logits, AZ).view(self.prompt_token_num, self.hidden_size)

        prompt_embedding = prompt_embedding.tile(batch_size, 1, 1)
        
        # a = torch.where(self.prompt_logits>0.5,self.ones,self.zeros)
        # print("a = " ,a)
        loss = self.neg_log_IBP(self.prompt_logits,self.n_tasks,self.n_prompts)
        return prompt_embedding,self.prompt_logits,loss


    def neg_log_IBP(self, matrix,n_tasks,n_prompts , alpha=3.):
        """ Calculate IBP prior contribution - log P(Z|alpha)
            Based on https://github.com/davidandrzej/PyIBP/blob/master/PyIBP.py """
        try:
            matrix = RelaxedBernoulli(temperature=1., logits= matrix).rsample()
        except ValueError:
            print(matrix)
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + self.EPS)
        # ones = torch.ones(n_tasks, n_prompts).to("cuda:0")
        # zeros = torch.zeros(n_tasks, n_prompts).to("cuda:0")
        # matrix = torch.where(matrix>0.5,ones,zeros)
        N, _ = matrix.shape
        # print("matrix.shape=",matrix.shape)
        # print("matrix=",matrix)
        m = matrix.sum(dim=0)
        # print("m.shape=",m.shape)
        # print("m=",m)
        # print("m.nonzero()=",m.nonzero())
        # print("m.nonzero() shape=",m.nonzero().shape)
        m = m[m.nonzero()].squeeze()
        # print("m.shape=",m.shape)
        K = len(m)
        # print(K)

        def log_factorial(value):
            return torch.lgamma(value + 1)

        logp = 0.
        logp += K * math.log(alpha)

        for n in range(N):
            new_features = torch.clamp(matrix[n] - matrix.sum(0), min=0., max=1.).sum()
            logp -= log_factorial(new_features)

        logp -= alpha * sum([float(1) / i for i in range(1, N + 1)])
        logp += (log_factorial(N - m) + log_factorial(m - 1)).sum()
        logp -= special.gammaln(N + 1) * K
        return - logp

# model = PromptChoice(d=500,hidden_size=5,prompt_token_num=3,n_tasks=6,n_prompts=2)
# task_id=torch.tensor([0])
# output = model(task_id=task_id,batch_size=4)
# print(output.shape)
# print(output)
