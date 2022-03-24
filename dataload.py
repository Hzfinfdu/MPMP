import copy
from datasets import load_dataset, concatenate_datasets
import torch
import random
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

torch.manual_seed(42)
random.seed(42)
tokenizer = BertTokenizer.from_pretrained("fnlp/cpt-large")


class TrainDataLoader:
    """
    封装了全部细节，只需要一直next，就可以保证每|NumOfDataset|个iter必每个任务都出现一遍，并且顺序随机。
    数据量长短的问题已在InfiniteDataLoader类中解决。
    """
    def __init__(self, batch_size=32, n_prompt_tokens=50):
        self.count = 0
        self.perm = torch.randperm(num_datasets)
        self.loader_list = [cls(n_prompt_tokens).get_infinite_dataloader(batch_size) for cls in Dataset_list]

    def __next__(self):
        next_batch = self.loader_list[self.perm[self.count]].__next__()
        # next_batch['task_id'] = self.perm[self.count].unsqueeze(0)
        self.count += 1
        if self.count == num_datasets:
            self.count = 0
            self.perm = torch.randperm(num_datasets)
        return next_batch, self.perm[self.count].unsqueeze(0)


def get_dataloaders(batch_size=32, split='validation'):
    dev_dataset = [cls().get_dataset(split) for cls in Dataset_list]
    return [
        torch.utils.data.DataLoader(ds, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate, pin_memory=True)
        for ds in dev_dataset
    ]


class InfiniteDataLoader:
    def __init__(self, *args, **kwargs):
        self.loader = torch.utils.data.DataLoader(*args, **kwargs)
        self.generator = self.loader.__iter__()

    def __next__(self):
        try:
            return self.generator.__next__()
        except StopIteration:
            self.generator._reset(self.loader)
            return self.generator.__next__()


def collate(batch_input):
    input_ids = [torch.tensor(d['input_ids']) for d in batch_input]
    start_positions = torch.tensor([d['start_positions'] for d in batch_input])
    end_positions = torch.tensor([d['end_positions'] for d in batch_input])
    input_ids = pad_sequence(input_ids, batch_first=True)
    return {
        'input_ids': input_ids,
        'start_positions': start_positions,
        'end_positions': end_positions
    }


class BasicDataset:
    def __init__(self, path, labellist, n_prompt_tokens):
        self.path = path
        self.labellist = labellist
        self.tmp_labellist = copy.deepcopy(labellist)
        self.has_test = False
        self.max_input_len = 511 - len('，'.join(labellist)) - n_prompt_tokens
        offset = 5000
        self.init_prompt = list(range(offset, offset + n_prompt_tokens))

    def convert_examples(self, example):
        input_ids = self.init_prompt + tokenizer.encode(self.input_template(example))[:-1][:self.max_input_len]
        random.shuffle(self.tmp_labellist)
        options = '，'.join(self.tmp_labellist)
        input_ids = input_ids
        start_positions = options.find(self.labellist[int(example['label'])]) + len(input_ids)
        return {
            'input_ids': input_ids + tokenizer.encode(options)[1:],
            'start_positions': start_positions,
            'end_positions': start_positions + len(self.labellist[int(example['label'])]) - 1
        }

    def input_template(self, example):
        raise NotImplementedError

    def get_dataset(self, split='train'):
        if not self.has_test:
            if split == 'train':
                dataset = load_dataset(self.path, split='train')
                dataset = dataset.train_test_split(test_size=.2, shuffle=False)['train']
            elif split == 'validation':
                dataset = load_dataset(self.path, split='train')
                dataset = dataset.train_test_split(test_size=.2, shuffle=False)['test']
            else:
                dataset = load_dataset(self.path, split='validation')
        else:
            dataset = load_dataset(self.path, split=split)
        return dataset.map(self.convert_examples, remove_columns=dataset.column_names)

    def get_infinite_dataloader(self, batch_size=32):
        return InfiniteDataLoader(self.get_dataset(), batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate)


class AFQMCDataset(BasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(path='/remote-home/share/ChineseData/chineseeval/AFQMC/AFQMC.py', labellist=["不同", "相似"], n_prompt_tokens=n_prompt_tokens)
        self.has_test = False

    def input_template(self, example):
        return f'意思判别：“{example["text1"]}”与“{example["text2"]}”的意思是？'



class OcnliDataset(BasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(path='/remote-home/share/ChineseData/chineseeval/ocnli/ocnli.py', labellist=["矛盾", "中立", "蕴含"], n_prompt_tokens=n_prompt_tokens)
        self.has_test = False

    def input_template(self, example):
        return f'意思判别：“{example["text1"]}”与“{example["text2"]}”的关系是？'

class PawsDataset(BasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(path='/remote-home/share/ChineseData/chineseeval/paws/paws.py', labellist=["矛盾", "中立", "蕴含"], n_prompt_tokens=n_prompt_tokens)
        self.has_test = True

    def input_template(self, example):
        return f'意思判别：“{example["text1"]}”与“{example["text2"]}”的关系是？'


class CMNLIDataset(BasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(path='/remote-home/share/ChineseData/chineseeval/CMNLI/cmnli.py', labellist=["矛盾", "中立", "蕴含"], n_prompt_tokens=n_prompt_tokens)
        self.has_test = False

    def input_template(self, example):
        return f'意思判别：“{example["text1"]}”与“{example["text2"]}”的关系是？'


class ChnSentiCorpDataset(BasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(path='/remote-home/share/ChineseData/chineseeval/chnsenticorp/chnsenticorp.py', labellist=["负面", "正面"], n_prompt_tokens=n_prompt_tokens)
        self.has_test = True

    def input_template(self, example):
        return f'情感分析：“{example["text"]}”的情感是？'
    

class THUCNewsDataset(BasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(path='/remote-home/share/ChineseData/chineseeval/THUCNews/thuc_news.py',
                         labellist=["体育", "娱乐", "财经", "教育", "时尚", "八卦", "游戏", "社会", "科技", "经济"], n_prompt_tokens=n_prompt_tokens)
        self.has_test = True

    def input_template(self, example):
        return f'主题识别：“{example["text"]}”的主题是？'


Dataset_list = [
    AFQMCDataset,
    # OcnliDataset,
    PawsDataset,
    # CMNLIDataset,
    ChnSentiCorpDataset,
    THUCNewsDataset,
]

num_datasets = len(Dataset_list)


def taskname2dataloader(taskname):
    if taskname not in Dic.keys():
        raise(ValueError("Taskname is Non-existent"))
    return Dic[taskname]().get_dataloader()


if __name__ == '__main__':
    a = TrainDataLoader()
    while True:
        batch, task_id = a.__next__()
        batch['task_id'] = task_id
        for k, v in batch.items():
            batch[k] = v.to('cuda:0')
