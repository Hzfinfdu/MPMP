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

    def forward(self, input_ids, start_positions, end_positions, task_id, label_mask=None, label=None, is_train=True):
        batch_size = input_ids.size()[0]
        prompt_embedding, prompt_logits, IBP_loss = self.prompt_embed_model(task_id=task_id, batch_size=batch_size,
                                                                            is_train=is_train)

        outputs = self.model(input_ids=input_ids, prompt_embedding=prompt_embedding, start_positions=start_positions,
                             end_positions=end_positions)
        loss = outputs.loss
        if label_mask is None:
            acc = torch.mul((start_positions == outputs.start_logits.argmax(dim=1)).long(),
                            (end_positions == outputs.end_logits.argmax(dim=1)).long()).sum() / batch_size
        else:
            # label_mask = torch.tensor(label_mask)
            bsz = label_mask.size(0)
            nz = label_mask.nonzero().t()
            start_logits = torch.mul(label_mask, outputs.start_logits)[nz[0], nz[1]].view(bsz, -1, 2)[:, :, 0]
            end_logits = torch.mul(label_mask, outputs.end_logits)[nz[0], nz[1]].view(bsz, -1, 2)[:, :, 1]
            probs = torch.mul(start_logits, end_logits)
            preds = probs.argmax(dim=1)
            acc = (preds == label).sum() / bsz
            print(acc)
        # print("######loss=",loss)
        # print("IBPloss=",IBP_loss)
        loss = loss + IBP_loss
        return IBP_loss, loss, acc, prompt_logits
        # loss = prompt_embedding.sum(dim=0).sum(dim=0).sum(dim=0)
        # return loss, torch.tensor(0)


class PromptChoice(nn.Module):
    def __init__(self, d, hidden_size, prompt_token_num, n_tasks, n_prompts):
        super(PromptChoice, self).__init__()
        self.prompt_logits = nn.Parameter(torch.empty((n_tasks, n_prompts)).uniform_(-1e-3, 1e-3))
        self.ones = nn.Parameter(torch.ones(n_tasks, n_prompts))
        self.zeros = nn.Parameter(torch.zeros(n_tasks, n_prompts))
        # self.Z = nn.Parameter(torch.rand(n_prompts, d, 1))
        # self.A = nn.Parameter(torch.rand(n_prompts, prompt_token_num * hidden_size, d))
        self.AZ = nn.Parameter(torch.rand(n_prompts, prompt_token_num * hidden_size))
        self.hidden_size = hidden_size
        self.prompt_token_num = prompt_token_num
        self.n_tasks = n_tasks
        self.n_prompts = n_prompts
        self.EPS = 1e-12
        self.temperature = 20.

    def forward(self, task_id, batch_size, is_train):
        prompt_logits = self.prompt_logits[task_id]
        try:
            prompt_logits = RelaxedBernoulli(temperature=self.temperature, logits=prompt_logits).rsample()
        except ValueError:
            print(prompt_logits)
        prompt_logits = prompt_logits / (prompt_logits.sum(dim=-1, keepdim=True) + self.EPS)
        # AZ = torch.bmm(self.A, self.Z).squeeze(-1)
        prompt_embedding = torch.mm(prompt_logits, self.AZ).view(self.prompt_token_num, self.hidden_size)

        prompt_embedding = prompt_embedding.tile(batch_size, 1, 1)

        loss = self.neg_log_IBP(self.prompt_logits)
        return prompt_embedding, self.prompt_logits, loss

    def neg_log_IBP(self, matrix, alpha=3.):
        """ Calculate IBP prior contribution - log P(Z|alpha)
            Based on https://github.com/davidandrzej/PyIBP/blob/master/PyIBP.py """
        N, _ = matrix.shape
        matrix = RelaxedBernoulli(torch.tensor(0.1), logits=matrix).sample()
        m = matrix.sum(dim=0)
        m_int = (matrix > 0.5).long()
        m_int = m_int.sum(dim=0)
        K = m_int.nonzero().size(0)
        m = m[m_int.nonzero()].squeeze()

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

    # def neg_log_IBP(self, matrix, alpha=3.):
    #     """ Calculate IBP prior contribution - log P(Z|alpha)
    #         Based on https://github.com/davidandrzej/PyIBP/blob/master/PyIBP.py """
    #     matrix = torch.sigmoid(matrix)
    #     matrix_hard = (matrix > .5).float()
    #     matrix = matrix_hard - matrix.detach() + matrix
    #
    #     N, _ = matrix.shape
    #     m = matrix.sum(dim=0)
    #     m = m[m.nonzero()].squeeze()
    #     K = len(m)
    #
    #     def log_factorial(value):
    #         return torch.lgamma(value + 1)
    #
    #     logp = 0.
    #     logp += K * math.log(alpha)
    #
    #     for n in range(N):
    #         new_features = torch.clamp(matrix[n] - matrix.sum(0), min=0., max=1.).sum()
    #         logp -= log_factorial(new_features)
    #
    #     logp -= alpha * sum([float(1) / i for i in range(1, N + 1)])
    #     logp += (log_factorial(N - m) + log_factorial(m - 1)).sum()
    #     logp -= special.gammaln(N + 1) * K
    #     return - logp
