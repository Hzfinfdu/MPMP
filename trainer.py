import copy
import numpy as np
from modeling_cpt import CPTForConditionalGeneration
import torch
from dataload import TrainDataLoader, get_dataloaders, num_datasets, tokenizer, Dataset_list
from tqdm import tqdm
import os
import fastNLP
import time
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


class MutitaskTrainer(object):
    def __init__(self, args, model, optimizer, scheduler=None):
        """
        :param model: 模型
        :param optimizer: 优化器
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
        self.n_prompt_tokens = args.n_prompt_tokens
        self.total_loss = 0.
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

    def _write_summary(self, *args):
        with open(os.path.join(self.save_path, 'logs.txt'), 'a+') as f:
            print(*args, file=f)

    def _write_router(self):
        with open(os.path.join(self.save_path, 'router.txt'), 'a+') as f:
            print(f' - Step {self.steps}: {self.model.prompt_embed_model.prompt_logits}', file=f)

    def _preview_datasets(self):
        for i in range(num_datasets):
            batch, task_id = next(self.train_loader)
            info_str = ('-----------------------------------------\n'
                        f'Dataset [{Dataset_list[task_id].__name__}] with task id [{task_id.item()}].\n'
                        f'An example: [{tokenizer.decode(batch["input_ids"][0][self.n_prompt_tokens + 1:])}]\n'
                        f'Its label is [{tokenizer.decode(batch["input_ids"][0][batch["start_positions"][0]: batch["end_positions"][0] + 1])}]\n'
                        '-----------------------------------------\n')
            self.logger.info(info_str)
            self._write_summary(info_str)

    def train(self):
        self._preview_datasets()
        for param in self.model.model.model.parameters():
            param.requires_grad = False
        for param in self.model.prompt_embed_model.parameters():
            param.requires_grad = True
        self.model.model.qa_outputs.weight.requires_grad = False
        self.model.to(self.device)
        total_time = time.time()
        self.logger.info("Start training...")
        for i_step in tqdm(range(self.n_steps)):
            self._train_step()
            if self.anneal_rate is not None and self.anneal_min is not None:
                self._anneal(self.steps)
            if i_step % self.eval_every == self.eval_every - 1:
                dev_loss, dev_acc = self._eval_epoch()
                mean_acc = sum(dev_acc) / len(dev_acc)
                self._dump_model_state(f"{self.steps}.th")
                eval_str = f"Validation loss {sum(dev_loss) / len(dev_loss)}, avg acc {mean_acc}"
                for task, value in enumerate(dev_acc):
                    eval_str += f", task {task} acc {value}"
                self.logger.info(eval_str)
                self._write_summary(eval_str)

                if mean_acc > self.best_acc:
                    self.best_acc = mean_acc
                    self.best_step = self.steps
                    self.logger.info("Updating best model...")
                    self._save_model()
                    self.logger.info("Model saved.")

                    self.logger.info(f"Current best acc [{self.best_acc}] occurred at step [{self.best_step}].")
        self.logger.info("Training finished. Elapse {:.4f} hours.".format((time.time() - total_time) / 3600))

    def _train_step(self):
        batch, task_id = next(self.train_loader)
        batch['task_id'] = task_id
        for k, v in batch.items():
            if batch[k] is not None:
                batch[k] = v.to(self.device)
        self.model.prompt_embed_model.train()
        self.model.model.model.eval()
        self.model.model.qa_outputs.train()
        self.model.zero_grad()
        loss, acc = self.model(**batch)
        self.total_loss += loss.item()
        self.steps += 1
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        if self.steps % self.print_every == 0:
            self._write_summary("train_loss", self.total_loss / self.print_every, self.steps)
            self._write_router()
            self.logger.info(f" - Step {self.steps}: router {self.model.prompt_embed_model.prompt_logits}")
            self.logger.info(f" - Step {self.steps}: loss {self.total_loss / self.print_every}")
            self.logger.info(f" - Step {self.steps}: temperature {self.model.prompt_embed_model.temperature}")
            self.total_loss = 0.
        if self.scheduler is not None:
            self.scheduler.step()

    def _eval_epoch(self):
        self.logger.info("Evaluating...")
        dev_losses = []
        dev_accs = []
        self.model.model.eval()
        self.model.prompt_embed_model.eval()
        with torch.no_grad():
            for id_, dev_loader in enumerate(self.dev_loaders):
                total_loss, total_acc = 0., 0.
                for i, batch in tqdm(enumerate(dev_loader)):
                    batch['task_id'] = torch.tensor([id_])
                    for k, v in batch.items():
                        if batch[k] is not None:
                            batch[k] = v.to(self.device)
                    batch['is_train'] = False
                    loss, acc = self.model(**batch)
                    total_loss += loss.item()
                    total_acc += acc
                total_loss /= len(dev_loader)
                total_acc /= len(dev_loader)
                dev_losses.append(total_loss)
                dev_accs.append(total_acc)
        return dev_losses, dev_accs

    def _save_model(self):
        save_path = os.path.join(self.save_path, "best.th")
        torch.save({
            'skilled_prompts': self.model.prompt_embed_model.state_dict(),
            'lmhead': self.model.model.qa_outputs.weight,
            'optimizer': self.optim.state_dict(),
        }, save_path)

    def _dump_model_state(self, name):
        save_path = os.path.join(self.save_path, "models", name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'skilled_prompts': self.model.prompt_embed_model.state_dict(),
            'lmhead': self.model.model.qa_outputs.weight,
            'optimizer': self.optim.state_dict(),
        }, save_path)

    def _anneal(self, i_step):
        self.model.prompt_embed_model.temperature = max(self.anneal_min,
                                                        self.model.prompt_embed_model.temperature * np.exp(
                                                            -self.anneal_rate * i_step))

    @classmethod
    def from_checkpoint(cls, args, model, optimizer, steps, scheduler=None):
        print('Recovering...')
        args.n_steps -= steps
        state = torch.load(os.path.join(args.save_path, 'models', str(steps) + '.th'))
        model.prompt_embed_model.load_state_dict(state['skilled_prompts'])
        model.model.qa_outputs.weight = state['lmhead']
        optimizer.load_state_dict(state['optimizer'])
        optimizer.cuda()
        trainer = cls(args, model, optimizer, scheduler)
        trainer.steps = steps
        if trainer.anneal_rate is not None and trainer.anneal_min is not None:
            for i in range(steps):
                trainer._anneal(i)
        trainer.model.to(trainer.device)
        dev_loss, dev_acc = trainer._eval_epoch()
        mean_acc = sum(dev_acc) / len(dev_acc)
        trainer.best_acc = mean_acc
        trainer.best_step = steps
        print('Recover finished')
        trainer._write_summary(f'Training recovered from step {steps}, performance at this step is {mean_acc}')
        return trainer
