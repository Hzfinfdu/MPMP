import json
import torch
import torch.nn as nn
import math
from scipy import special
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from transformers import BertTokenizerFast
from modeling_cpt import CPTForConditionalGeneration, CPTForQuestionAnswering
from transformers import BertTokenizer
import datasets

# n_tasks = 6
# n_prompts = 2
# prompt_token_num = 3
# d = 500
# D = prompt_token_num*4096
# Taks2Prompt = torch.rand(Task_num,prompt_token_num)

class PretrainPrompt(nn.Module):
    def __init__(self, d, prompt_token_num, n_tasks, n_prompts, init_temperature):
        super(PretrainPrompt, self).__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained("fnlp/cpt-large")
        self.metric = datasets.load_metric("squad")
        prefix_config = {
            'n_prompt_tokens': prompt_token_num,
            'n_tasks': n_tasks,
            'n_prompts': n_prompts,
            'temperature': init_temperature,
        }
        self.model = CPTForQuestionAnswering.from_pretrained("fnlp/cpt-large", prefix_config=prefix_config)

    def forward(self, input_ids, start_positions, end_positions, task_id, label_mask=None, label=None, is_train=True):
        batch_size = input_ids.size(0)
        self.inform_model(task_id)
        outputs = self.model(input_ids=input_ids, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        if is_train:
            acc = None
        else:
            if label_mask is None:
                pred_char_span = []
                gold_char_span = []
                for i in range(batch_size):
                    pred_char_span.append({
                        "id": str(i),
                        'prediction_text': self.tokenizer.decode(
                            input_ids[i, outputs.start_logits.argmax(dim=1)[i]:outputs.end_logits.argmax(dim=1)[i] + 1])
                    })
                    gold_char_span.append({
                        "id": str(i),
                        'answers': {
                            'text': [self.tokenizer.decode(input_ids[i, start_positions[i]:end_positions[i] + 1])],
                            'answer_start': [0]
                        },
                    })
                acc = self.metric.compute(predictions=pred_char_span, references=gold_char_span)['f1'] / 100.
            else:
                nz = (label_mask.nonzero().reshape(-1, 4) - torch.LongTensor([0, 0, 0, 1]).cuda()).reshape(-1, 2).t()
                start_logits = outputs.start_logits[nz[0], nz[1]].view(batch_size, -1, 2)[:, :, 0]
                end_logits = outputs.end_logits[nz[0], nz[1]].view(batch_size, -1, 2)[:, :, 1]
                probs = torch.mul(start_logits, end_logits)
                preds = probs.argmax(dim=1)
                acc = (preds == label).sum() / batch_size
        return loss, acc

    def inform_model(self, task_id):
        self.model.model.encoder.encoder.task_id = task_id


