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
        batch_size = input_ids.size(0)
        prompt_embedding = self.prompt_embed_model(task_id=task_id, batch_size=batch_size, is_train=is_train)

        outputs = self.model(input_ids=input_ids, prompt_embedding=prompt_embedding, start_positions=start_positions,
                             end_positions=end_positions)
        loss = outputs.loss
        if is_train:
            acc = None
        else:
            if label_mask is None:
                acc = torch.mul((start_positions == outputs.start_logits.argmax(dim=1)).long(),
                                (end_positions == outputs.end_logits.argmax(dim=1)).long()).sum() / batch_size
            else:
                # label_mask = torch.tensor(label_mask)
                nz = (label_mask.nonzero().view(-1, 4) - torch.LongTensor([0, 0, 0, 1], device=label_mask.device)).view(-1, 2).t()
                start_logits = outputs.start_logits[nz[0], nz[1]].view(batch_size, -1, 2)[:, :, 0]
                end_logits = outputs.end_logits[nz[0], nz[1]].view(batch_size, -1, 2)[:, :, 1]
                probs = torch.mul(start_logits, end_logits)
                preds = probs.argmax(dim=1)
                acc = (preds == label).sum() / batch_size
            # print("######loss=",loss)
        return loss, acc
        # loss = prompt_embedding.sum(dim=0).sum(dim=0).sum(dim=0)
        # return loss, torch.tensor(0)


class PromptChoice(nn.Module):
    def __init__(self, d, hidden_size, prompt_token_num, n_tasks, n_prompts):
        super(PromptChoice, self).__init__()
        self.prompt_logits = nn.Parameter(torch.empty((n_tasks, n_prompts)).uniform_(-1e-3, 1e-3))
        # self.Z = nn.Parameter(torch.rand(n_prompts, d, 1))
        # self.A = nn.Parameter(torch.rand(n_prompts, prompt_token_num * hidden_size, d))
        self.AZ = nn.Parameter(torch.empty((n_prompts, prompt_token_num * hidden_size)).uniform_(-1e-3, 1e-3))
        self.hidden_size = hidden_size
        self.prompt_token_num = prompt_token_num
        self.n_tasks = n_tasks
        self.n_prompts = n_prompts
        self.EPS = 1e-12
        # self.ibp_alpha = ibp_alpha
        # self.ibp_omg = ibp_omg

    def forward(self, task_id, batch_size, is_train):
        prompt_logits = self.prompt_logits[task_id]
        # try:
        if is_train:
            prompt_logits = RelaxedBernoulli(temperature=1., logits=prompt_logits).rsample()
        else:
            prompt_logits = torch.sigmoid(prompt_logits)

        # except ValueError:
        #     print(prompt_logits)
        # mask = (prompt_logits > .5).float()
        # prompt_logits = torch.mul(mask, prompt_logits)
        prompt_logits = prompt_logits / (prompt_logits.sum(dim=-1, keepdim=True) + self.EPS)
        # AZ = torch.bmm(self.A, self.Z).squeeze(-1)
        prompt_embedding = torch.mm(prompt_logits, self.AZ).view(self.prompt_token_num, self.hidden_size)

        prompt_embedding = prompt_embedding.tile(batch_size, 1, 1)

        return prompt_embedding


