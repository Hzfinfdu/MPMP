import copy
import numpy as np
from modeling_cpt import CPTForConditionalGeneration
import torch
from dataload import *
from tqdm import tqdm
import os
import fastNLP
from transformers import BertTokenizerFast
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
        self.accumulation_step = args.accumulation_step
        self.train_loader = get_infinite_train_iterator(self.batch_size, args.n_prompt_tokens)
        self.dev_loaders = get_dataloaders(batch_size=self.batch_size, split='validation')
        self.tokenizer = BertTokenizerFast.from_pretrained("fnlp/cpt-large")
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
            print(f' - Step {self.steps}: {self.model.model.model.encoder.encoder.router}', file=f)

    def _preview_datasets(self):
        for i in range(num_datasets):
            batch, task_id = next(self.train_loader)[0]
            info_str = ('-----------------------------------------\n'
                        f'Dataset [{Dataset_list[task_id].__name__}] with task id [{task_id}].\n'
                        f'An example: [{self.tokenizer.decode(batch["input_ids"][0][self.n_prompt_tokens + 1:])}]\n'
                        f'Its label is [{self.tokenizer.decode(batch["input_ids"][0][batch["start_positions"][0]: batch["end_positions"][0] + 1])}]\n'
                        '-----------------------------------------\n')
            self.logger.info(info_str)
            self._write_summary(info_str)

    def train(self):
        self._preview_datasets()
        for param in self.model.model.model.parameters():
            param.requires_grad = False
        self.model.model.model.encoder.encoder.router.requires_grad = True
        self.model.model.model.encoder.encoder.A.requires_grad = True
        self.model.model.model.encoder.encoder.z.requires_grad = True
        self.model.model.qa_outputs.requires_grad = True
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
        batch, task_id = next(self.train_loader)[0]
        batch['task_id'] = torch.tensor([task_id])
        for k, v in batch.items():
            if batch[k] is not None:
                batch[k] = v.to(self.device)
        self.model.model.train()
        loss, acc = self.model(**batch)
        self.total_loss += loss.item()
        self.steps += 1
        loss.backward()
        if self.steps % self.print_every == 0:
            self._write_summary("train_loss", self.total_loss / self.print_every, self.steps)
            # self._write_router()
            self.logger.info(f" - Step {self.steps}: router {self.model.model.model.encoder.encoder.router}")
            self.logger.info(f" - Step {self.steps}: loss {self.total_loss / self.print_every}")
            self.total_loss = 0.
        if self.steps % self.accumulation_step == 0:
            self.optim.step()
            self.optim.zero_grad()
            self.model.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()

    def _eval_epoch(self):
        self.logger.info("Evaluating...")
        dev_losses = []
        dev_accs = []
        self.model.model.eval()
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
            'A': self.model.model.model.encoder.encoder.A,
            'z': self.model.model.model.encoder.encoder.z,
            'router': self.model.model.model.encoder.encoder.router,
            'lmhead': self.model.model.qa_outputs.weight,
            'optimizer': self.optim.state_dict(),
        }, save_path)

    def _dump_model_state(self, name):
        save_path = os.path.join(self.save_path, "models", name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'A': self.model.model.model.encoder.encoder.A,
            'z': self.model.model.model.encoder.encoder.z,
            'router': self.model.model.model.encoder.encoder.router,
            'lmhead': self.model.model.qa_outputs.weight,
            'optimizer': self.optim.state_dict(),
        }, save_path)

    # def _anneal(self, i_step):
    #     self.model.prompt_embed_model.temperature = max(self.anneal_min,
    #                                                     self.model.prompt_embed_model.temperature * np.exp(
    #                                                         -self.anneal_rate * i_step))

    @classmethod
    def from_checkpoint(cls, args, model, Optim, steps, scheduler=None):
        print('Recovering...')
        args.n_steps -= steps
        state = torch.load(os.path.join(args.save_path, 'models', str(steps) + '.th'))
        model.model.model.encoder.encoder.A = state['A']
        model.model.model.encoder.encoder.z = state['z']
        model.model.model.encoder.encoder.router = state['router']
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
        optimizer.load_state_dict(state['optimizer'])
        optimizer.cuda()
        trainer = cls(args, model, optimizer, scheduler)
        trainer.steps = steps
        if trainer.anneal_rate is not None and trainer.anneal_min is not None:
            for i in range(steps):
                trainer._anneal(i)
        trainer.model.to(trainer.device)
        # dev_loss, dev_acc = trainer._eval_epoch()
        # mean_acc = sum(dev_acc) / len(dev_acc)
        # trainer.best_acc = mean_acc
        # trainer.best_step = steps
        # print('Recover finished')
        # trainer._write_summary(f'Training recovered from step {steps}, performance at this step is {mean_acc}')
        return trainer


class DownstreamTrainer:
    dataloaders = {
        'chnsenticorp': ChnSentiCorpDataset(),
        'iflytek': iflytekDataset(),
        'lcqmc': LCQMCDataset(),
    }
    def __init__(self, args, model, optimizer, scheduler=None):
        """
        :param model: 模型
        :param optimizer: 优化器
        :param save_path: 模型存储位置
        """
        self.logger = fastNLP.logger
        self.save_path = args.save_path
        self.optim = optimizer
        # self.n_steps = args.n_steps
        self.n_epochs = args.n_epochs
        self.scheduler = scheduler
        self.eval_every = args.eval_every
        self.batch_size = args.batch_size
        self.save_every = args.save_every
        self.print_every = args.print_every
        self.anneal_rate = args.anneal_rate
        self.anneal_min = args.anneal_min
        self.n_prompt_tokens = args.n_prompt_tokens
        self.total_loss = 0.
        self.epochs = 0
        self.best_acc = 0
        self.best_epoch = 0
        self.seed = args.seed
        self.model = model
        self.device = args.device
        ds = self.dataloaders[args.task_name]
        data = ds.get_dataset(split='downstream', k_shot=args.k_shot, seed=self.seed)
        train_data = data['train']
        eval_data = data['dev']
        print(train_data.__len__())
        print(eval_data.__len__())
        test_data = ds.get_dataset(split='test')
        print(test_data.__len__())
        self.trainloader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True,
                                      collate_fn=self._collate)
        self.evalloader = torch.utils.data.DataLoader(eval_data, batch_size=self.batch_size, shuffle=False,
                                     collate_fn=self._collate)
        self.testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False,
                                     collate_fn=self._collate)
        self.logger.info(
            '-------------Trainer info-------------\n'
            f'Save path {self.save_path}\n'
            f'Number of epochs {self.n_epochs}\n'
            f'Batch size {self.batch_size}\n'
            f'Scheduler {self.scheduler}\n'
            f'Saves every {self.save_every} steps\n'
            '---------End of Trainer info----------\n'
        )

    @staticmethod
    def _collate(batch_input):
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
            'label': label
        }

    # def _write_summary(self, *args):
    #     with open(os.path.join(self.save_path, 'downstream_logs.txt'), 'a+') as f:
    #         print(*args, file=f)

    def train(self):
        for param in self.model.model.model.parameters():
            param.requires_grad = False
        for param in self.model.model.model.parameters():
            param.requires_grad = False
        self.model.model.model.encoder.encoder.router.requires_grad = True
        self.model.model.model.encoder.encoder.A.requires_grad = True
        self.model.model.model.encoder.encoder.z.requires_grad = True
        self.model.model.qa_outputs.requires_grad = True
        self.model.to(self.device)
        total_time = time.time()
        self.logger.info("Start training...")
        for i_epoch in tqdm(range(self.n_epochs)):
            self.total_loss = 0.
            n_batchs = 0
            for i, iter in enumerate(self.trainloader):
                for k, v in iter.items():
                    if iter[k] is not None:
                        iter[k] = v.to(self.device)
                self.model.model.train()
                # self.model.model.model.eval()
                # self.model.model.qa_outputs.eval()
                self.model.model.eval()
                self.model.zero_grad()
                loss, acc = self.model(**iter)
                self.total_loss += loss.item()
                # self.steps += 1
                n_batchs += 1
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()
            self.epochs += 1
            # self._write_summary("train_loss", self.total_loss / n_batchs, i_epoch + 1)
            if i_epoch % self.eval_every == self.eval_every - 1:
                # self.logger.info(self.model.model.model.encoder.encoder.router)
                dev_loss, dev_acc = self._eval_epoch()
                eval_str = f"loss {dev_loss}, acc {dev_acc}"
                # self._write_summary(eval_str)

                if dev_acc > self.best_acc:
                    self.best_acc = dev_acc
                    self.best_epoch = i_epoch
                    self.logger.info("Updating best model...")
                    self._save_model()
                    self.logger.info("Model saved.")
                    self.logger.info(f"Current best acc [{self.best_acc}] occurred at step [{self.best_epoch}].")
            if i_epoch == self.n_epochs - 1: print(
                f"Current best acc [{self.best_acc}] occurred at step [{self.best_epoch}].")
        state = torch.load(os.path.join(self.save_path, "best.th"))
        self.model.model.model.encoder.encoder.A = state['A']
        self.model.model.model.encoder.encoder.z = state['z']
        self.model.model.model.encoder.encoder.router = state['router']
        self.model.model.qa_outputs.weight = state['lmhead']
        test_loss, test_acc = self._test_epoch()
        test_str = f"test loss {test_loss}, acc {test_acc}"
        self.logger.info(test_str)
        self.logger.info("Training finished. Elapse {:.4f} hours.".format((time.time() - total_time) / 3600))
        return test_acc

    def _eval_epoch(self):
        self.model.model.eval()
        with torch.no_grad():
            total_loss, total_acc, n_batchs = 0., 0., 0
            for i, iter in enumerate(self.evalloader):
                for k, v in iter.items():
                    if iter[k] is not None:
                        iter[k] = v.to(self.device)
                iter['is_train'] = False
                loss, acc = self.model(**iter)
                total_loss += loss.item()
                total_acc += acc
                n_batchs += 1
            total_loss /= n_batchs
            total_acc /= n_batchs
        return total_loss, total_acc

    def _test_epoch(self):
        self.logger.info("Evaluating...")
        self.model.model.eval()
        with torch.no_grad():
            total_loss, total_acc, n_batchs = 0., 0., 0
            for i, iter in tqdm(enumerate(self.testloader)):
                for k, v in iter.items():
                    if iter[k] is not None:
                        iter[k] = v.to(self.device)
                iter['is_train'] = False
                loss, acc = self.model(**iter)
                total_loss += loss.item()
                total_acc += acc
                n_batchs += 1
            total_loss /= n_batchs
            total_acc /= n_batchs
        return total_loss, total_acc

    def _save_model(self):
        save_path = os.path.join(self.save_path, "best.th")
        torch.save({
            'A': self.model.model.model.encoder.encoder.A,
            'z': self.model.model.model.encoder.encoder.z,
            'router': self.model.model.model.encoder.encoder.router,
            'lmhead': self.model.model.qa_outputs.weight,
            'optimizer': self.optim.state_dict(),
        }, save_path)
