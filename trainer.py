import copy

from modeling_cpt import CPTForConditionalGeneration
from transformers import BertTokenizer
import torch
from dataload import TrainDataLoader, get_dataloaders
from tqdm import tqdm
import os
import fastNLP
import time


# tokenizer = BertTokenizer.from_pretrained("fnlp/cpt-large")
# model = CPTForConditionalGeneration.from_pretrained("fnlp/cpt-large")
#
# d = 500
# D = 1024
# Task_num = 6
# prompt_num = 2
# Taks2Prompt = torch.rand(Task_num,prompt_num)
# listZ = [torch.rand(d) for i in range(prompt_num)]
# listA = [torch.rand(d,D) for i in range(prompt_num)]
#

def write_summary(*args):
    with open('./logs/logs.txt', 'a+') as f:
        print(*args, file=f)


class MutitaskTrainer(object):
    def __init__(self, args, model, optimizer, scheduler=None):
        """
        :param model: 模型
        :param optimizer: 优化器
        :param log_path: TensorboardX存储文件夹
        :param save_path: 模型存储位置
        :param accumulation_steps: 累积梯度
        :param print_every: 评估间隔
        """
        self.logger = fastNLP.logger
        self.save_path = args.save_path
        self.optim = optimizer
        self.n_steps = args.n_steps
        self.scheduler = scheduler
        self.eval_every = args.eval_every
        self.batch_size = args.batch_size
        self.save_every = args.save_every
        self.print_every = args.print_every
        self.anneal_rate = args.anneal_rate
        self.anneal_min = args.anneal_min
        self.steps = 0
        self.best_acc = 0
        self.best_step = 0
        self.model = model
        self.device = args.device
        self.train_loader = TrainDataLoader(self.batch_size)
        self.dev_loaders = get_dataloaders(batch_size=self.batch_size, split='validation')
        self.logger.info(
            '-------------Trainer info-------------\n'
            f'Save path {self.save_path}\n'
            f'Number of steps {self.n_steps}\n'
            f'Batch size {self.batch_size}\n'
            f'Scheduler {self.scheduler}\n'
            f'Saves every {self.save_every} steps\n'
            '---------End of Trainer info----------\n'
        )
        if not os.path.exists('./logs'):
            os.makedirs('./logs', exist_ok=True)

    def train(self):
        for param in self.model.model.parameters():
            param.requires_grad = False
        for param in self.model.prompt_embed_model.parameters():
            param.requires_grad = True
        self.model.to(self.device)
        total_time = time.time()
        self.logger.info("Start training...")
        for i_step in tqdm(range(self.n_steps)):
            self._train_step()
            self.anneal(i_step)
            if i_step % self.eval_every == self.eval_every - 1:
                dev_loss, dev_acc = self._eval_epoch()
                mean_acc = sum(dev_acc) / len(dev_acc)
                self._dump_model_state(f"{i_step}.th")
                eval_str = f"Validation loss {sum(dev_loss) / len(dev_loss)}, avg acc {mean_acc}"
                for task, value in enumerate(dev_acc):
                    eval_str += f", task {task} acc {value}"
                self.logger.info(eval_str)

                if mean_acc > self.best_acc:
                    self.best_acc = mean_acc
                    self.best_step = i_step
                    self.logger.info("Updating best model...")
                    self._save_model()
                    self.logger.info("Model saved.")

                    self.logger.info(f"Current best acc [{self.best_acc}] occured at step [{self.best_step}].")
        self.logger.info("Training finished. Elapse {:.4f} hours.".format((time.time() - total_time) / 3600))

    def _train_step(self):
        batch, task_id = next(self.train_loader)
        batch['task_id'] = task_id
        for k, v in batch.items():
            batch[k] = v.to(self.device)
        self.model.prompt_embed_model.train()
        self.model.model.eval()
        self.model.zero_grad()
        loss, acc, prompt_logits = self.model(**batch)
        self.steps += 1
        if self.steps % 1000 == 0: print(prompt_logits)

        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        # print(f'step {self.steps}', torch.cuda.memory_summary())
        if self.steps % self.print_every == 0:
            write_summary("train_loss", loss.item() / self.print_every, self.steps)
            write_summary("train_acc", acc.item(), self.steps)
            self.logger.info(f" - Step {self.steps}: loss {loss.item() / self.print_every}")
        if self.scheduler is not None:
            self.scheduler.step()

    def _eval_epoch(self):
        self.logger.info("Evaluating...")
        dev_losses = []
        dev_accs = []
        self.model.eval()
        with torch.no_grad():
            for id_, dev_loader in enumerate(self.dev_loaders):
                total_loss, total_acc = 0., 0.
                for i, batch in tqdm(enumerate(dev_loader)):
                    batch['task_id'] = torch.tensor([id_])
                    for k, v in batch.items():
                        batch[k] = v.to(self.device)
                    batch['is_train'] = False
                    loss, acc, _ = self.model(**batch)
                    total_loss += loss.item()
                    total_acc += acc
                total_loss /= len(dev_loader)
                total_acc /= len(dev_loader)
                dev_losses.append(total_loss)
                dev_accs.append(total_acc)
        return dev_losses, dev_accs

    def _save_model(self):
        save_path = os.path.join(self.save_path, "best.th")
        torch.save(self.model.prompt_embed_model.state_dict(), save_path)

    def _dump_model_state(self, name):
        save_path = os.path.join(self.save_path, "models", name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.prompt_embed_model.state_dict(), save_path)

    def anneal(self, i_step):
        self.model.prompt_embed_model.temperature = max(self.anneal_min, self.model.prompt_embed_model.temperature * np.exp(-self.anneal_rate * i_step))
