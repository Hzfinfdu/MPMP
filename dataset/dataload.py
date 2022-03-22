from datasets import load_dataset
import CMNLI.CMNLI as data
# from transformers import BertTokenizer
import torch

# tokenizer = BertTokenizer.from_pretrained("fnlp/cpt-large")

raw_datasets = load_dataset(data.__file__)
print(raw_datasets)
print(raw_datasets['train'][0])


torch.manual_seed(42)
num2char = ['A','B','C','D','E','F','G','H','I','J']



class AFQMC:
    def convert_examples(self,example):
        choice_num = 2
        perm = torch.randperm(choice_num)
        lablelist = ["不同","相似"]
        example['input_text'] = f'问题：“{example["text1"]}”与“{example["text2"]}”的关系？选择：A. {lablelist[perm[0]]}.B.{lablelist[perm[1]]}.答案是[MASK]。'
        label_num = int(example['label'])
        x = (perm==label_num).nonzero(as_tuple=True)[0]
        example['answer'] = f'{num2char[x]}' 
        return example
    def get_dataset(self):
        dataset = load_dataset("/remote-home/qzhu/prompt_pretrain/dataset/AFQMC/AFQMC.py")
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        return dataset

class ocnli:
    def convert_examples(self,example):
        choice_num = 3
        perm = torch.randperm(choice_num)
        lablelist = ["矛盾","中立","蕴含"]
        example['input_text'] = f'问题：“{example["text1"]}”与“{example["text2"]}”的关系？选择：A. {lablelist[perm[0]]}.B.{lablelist[perm[1]]}.C.{lablelist[perm[2]]}.答案是[MASK]。'
        label_num = int(example['label'])
        x = (perm==label_num).nonzero(as_tuple=True)[0]
        example['answer'] = f'{num2char[x]}' 
        return example
    def get_dataset(self):
        dataset = load_dataset("/remote-home/qzhu/prompt_pretrain/dataset/ocnli/ocnli.py")
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        return dataset

class paws:
    def convert_examples(self,example):
        choice_num = 3
        perm = torch.randperm(choice_num)
        lablelist = ["矛盾","中立","蕴含"]
        example['input_text'] = f'问题：“{example["text1"]}”与“{example["text2"]}”的关系？选择：A. {lablelist[perm[0]]}.B.{lablelist[perm[1]]}.C.{lablelist[perm[2]]}.答案是[MASK]。'
        label_num = int(example['label'])
        x = (perm==label_num).nonzero(as_tuple=True)[0]
        example['answer'] = f'{num2char[x]}' 
        return example
    def get_dataset(self):
        dataset = load_dataset("/remote-home/qzhu/prompt_pretrain/dataset/paws/paws.py")
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        return dataset

class CMNLI:
    def convert_examples(self,example):
        choice_num = 3
        perm = torch.randperm(choice_num)
        lablelist = ["矛盾","中立","蕴含"]
        example['input_text'] = f'问题：“{example["text1"]}”与“{example["text2"]}”的关系？选择：A. {lablelist[perm[0]]}.B.{lablelist[perm[1]]}.C.{lablelist[perm[2]]}.答案是[MASK]。'
        label_num = int(example['label'])
        x = (perm==label_num).nonzero(as_tuple=True)[0]
        example['answer'] = f'{num2char[x]}' 
        return example
    def get_dataset(self):
        dataset = load_dataset("/remote-home/qzhu/prompt_pretrain/dataset/CMNLI/CMNLI.py")
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        return dataset        

class chnsenticorp:
    def convert_examples(self,example):
        choice_num = 2
        perm = torch.randperm(choice_num)
        lablelist = ["负面","正面"]
        example['input_text'] = f'问题：“{example["text"]}”的情感？选择：A. {lablelist[perm[0]]}.B.{lablelist[perm[1]]}.答案是[MASK]。'
        label_num = int(example['label'])
        x = (perm==label_num).nonzero(as_tuple=True)[0]
        example['answer'] = f'{num2char[x]}' 
        return example
    def get_dataset(self):
        dataset = load_dataset("/remote-home/qzhu/prompt_pretrain/dataset/chnsenticorp/chnsenticorp.py")
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        return dataset

class THUCNews:
    def convert_examples(self,example):
        choice_num = 10
        perm = torch.randperm(choice_num)
        lablelist = ["体育","娱乐","财经","教育","时尚","八卦","游戏","社会","科技","经济"]
        example['input_text'] = f'问题：“{example["text"]}”的主题？选择：A. {lablelist[perm[0]]}.B.{lablelist[perm[1]]}.C.{lablelist[perm[2]]}.D.{lablelist[perm[3]]}.E.{lablelist[perm[4]]}.F.{lablelist[perm[5]]}.G.{lablelist[perm[6]]}.H.{lablelist[perm[7]]}.I.{lablelist[perm[8]]}.J.{lablelist[perm[9]]}.答案是[MASK]。'
        label_num = int(example['label'])
        x = (perm==label_num).nonzero(as_tuple=True)[0]
        example['answer'] = f'{num2char[x]}' 
        return example
    def get_dataset(self):
        dataset = load_dataset("/remote-home/qzhu/prompt_pretrain/dataset/THUCNews/THUCNews.py")
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        return dataset

class fin_re:
    def convert_examples(self,example):
        choice_num = 44
        perm = torch.randperm(choice_num)
        lablelist = ['unknown',
                    '注资',
                    '拥有',
                    '纠纷',
                    '自己',
                    '增持',
                    '重组',
                    '买资',
                    '签约',
                    '持股',
                    '交易',
                    '入股',
                    '转让',
                    '成立',
                    '分析',
                    '合作',
                    '帮助',
                    '发行',
                    '商讨',
                    '合并',
                    '竞争',
                    '订单',
                    '减持',
                    '合资',
                    '收购',
                    '借壳',
                    '欠款',
                    '被发行',
                    '被转让',
                    '被成立',
                    '被注资',
                    '被持股',
                    '被拥有',
                    '被收购',
                    '被帮助',
                    '被借壳',
                    '被买资',
                    '被欠款',
                    '被增持',
                    '拟收购',
                    '被减持',
                    '被分析',
                    '被入股',
                    '被拟收购']
        example['input_text'] = f'问题：“{example["text"]}”的主题？选择：A. {lablelist[perm[0]]}.B.{lablelist[perm[1]]}.C.{lablelist[perm[2]]}.D.{lablelist[perm[3]]}.E.{lablelist[perm[4]]}.F.{lablelist[perm[5]]}.G.{lablelist[perm[6]]}.H.{lablelist[perm[7]]}.I.{lablelist[perm[8]]}.J.{lablelist[perm[9]]}.答案是[MASK]。'
        label_num = int(example['label'])
        x = (perm==label_num).nonzero(as_tuple=True)[0]
        example['answer'] = f'{num2char[x]}' 
        return example
    def get_dataset(self):
        dataset = load_dataset("/remote-home/qzhu/prompt_pretrain/dataset/fin_re/fin_re.py")
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        return dataset


Dic = {'AFQMC': AFQMC,'ocnli':ocnli,'paws':paws,'CMNLI':CMNLI,'chnsenticorp':chnsenticorp,'THUCNews':THUCNews}
##  CMNLI not finish yet. CMNLI.py exists bug.

def taskname2dataset(taskname):
    if taskname not in Dic.keys():
        raise(ValueError("Taskname is Non-existent"))
    return Dic[taskname]().get_dataset()
# a = AFQMC()
# print(a.get_dataset()['train'][0])
# a = taskname2dataset('AFQMC')
# a = taskname2dataset('CMNLI')
# print(a['train'][0])
