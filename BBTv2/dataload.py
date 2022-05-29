from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch
from transformers import BertTokenizerFast
from torch.nn.utils.rnn import pad_sequence



def get_infinite_train_iterator(bsz=32, n_prompt_tokens=50):
    loader = torch.utils.data.DataLoader(InfiniteDataset(bsz, n_prompt_tokens), batch_size=1, shuffle=True, num_workers=2, collate_fn=lambda x: x)
    return iter(loader)

def get_dataloaders(batch_size=32, split='validation'):
    dev_dataset = [cls().get_dataset(split) for cls in Dataset_list]

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
            'label': label
        }

    return [
        torch.utils.data.DataLoader(ds, batch_size=batch_size, drop_last=True, shuffle=False,
                                    collate_fn=collate, num_workers=4)
        for ds in dev_dataset
    ]


class InfiniteDataset(torch.utils.data.Dataset):
    def __init__(self, batch_size=32, n_prompt_tokens=50):
        self.ds_list = [cls(n_prompt_tokens).get_dataset() for cls in Dataset_list]
        self.len_list = [len(ds) // batch_size for ds in self.ds_list]
        self.batch_size = batch_size
        self.num_ds = len(Dataset_list)

    def __len__(self):
        return 1000000000  # infinity

    def __getitem__(self, idx):
        ds_idx = idx % self.num_ds
        batch_idx = (idx // self.num_ds) % self.len_list[ds_idx]
        data = self.ds_list[ds_idx][batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        return self.collate(data), ds_idx

    @staticmethod
    def collate(batch_input):
        input_ids = [torch.tensor(i) for i in batch_input['input_ids']]
        start_positions = torch.tensor(batch_input['start_positions'])
        end_positions = torch.tensor(batch_input['end_positions'])
        input_ids = pad_sequence(input_ids, batch_first=True)
        label_mask = None
        label = None
        if 'label_mask' in batch_input.keys():
            label_mask = [torch.tensor(i) for i in batch_input['label_mask']]
            label_mask = pad_sequence(label_mask, batch_first=True)
            assert label_mask.shape == input_ids.shape
            label = torch.tensor(batch_input['label'])
        return {
            'input_ids': input_ids,
            'start_positions': start_positions,
            'end_positions': end_positions,
            'label_mask': label_mask,
            'label': label
        }


class BasicDataset:
    offset = 1000
    tokenizer = BertTokenizerFast.from_pretrained("fnlp/cpt-large")
    data_dir = '/home/ma-user/work/zfhe/chineseeval'

    def __init__(self, path, has_test=False, n_prompt_tokens=50):
        self.path = path
        self.has_test = has_test
        self.init_prompt = list(range(self.offset, self.offset + n_prompt_tokens))
        self.n_prompt_tokens = n_prompt_tokens

    def input_template(self, example):
        raise NotImplementedError

    def convert_examples(self, example):
        raise NotImplementedError

    def get_dataset(self, split='train'):
        return self._get_dataset(split)

    def _get_dataset(self, split='train'):
        if not self.has_test:
            if split == 'train':
                dataset = load_dataset(self.path, split='train')
                test_size = .1 if len(dataset) < 122880 else 12288
                dataset = dataset.train_test_split(test_size=test_size, shuffle=True, seed=42)['train']
            elif split == 'validation':
                dataset = load_dataset(self.path, split='train')
                test_size = .1 if len(dataset) < 122880 else 12288
                dataset = dataset.train_test_split(test_size=test_size, shuffle=True, seed=42)['test']
            elif split == 'test':
                dataset = load_dataset(self.path, split='validation')
            elif split == 'downstream':
                dataset = load_dataset(self.path, split='downstream')
            else:
                raise ValueError(f'split `{split}` not supported')
        else:
            dataset = load_dataset(self.path, split=split)
        return dataset.map(self.convert_examples, remove_columns=dataset.column_names)


class TCNLIBasicDataset(BasicDataset):
    def __init__(self, path, n_prompt_tokens, has_test, labellist, label_mask):
        super(TCNLIBasicDataset, self).__init__(path, has_test, n_prompt_tokens)
        self.labellist = labellist
        self.option_ids = [self.tokenizer.encode(opt)[1:-1] for opt in labellist]
        self.option_lens = [len(opt) for opt in self.option_ids]
        self.option_starts = [0]
        for i in range(1, len(labellist)):
            self.option_starts.append(self.option_starts[i - 1] + 1 + self.option_lens[i - 1])
        self.max_input_len = 511 - sum(self.option_lens) - len(labellist) - n_prompt_tokens
        self.label_mask = label_mask
        self.option_ids_list = []
        for i in self.option_ids:
            self.option_ids_list.extend(i + [8024])

    def convert_examples(self, example):
        input_ids = [101] + self.init_prompt + self.tokenizer.encode(self.input_template(example))[1:-1][:self.max_input_len]
        input_ids_len = len(input_ids)
        start_positions = self.option_starts[int(example['label'])] + input_ids_len
        return {
            'input_ids': input_ids + self.option_ids_list,
            'start_positions': start_positions,
            'end_positions': start_positions + len(self.labellist[int(example['label'])]) - 1,
            'label_mask': [0] * input_ids_len + self.label_mask + [1],
            'label': int(example['label'])
        }

    def get_dataset(self, split='train', k_shot=32, seed=42):
        dataset = self._get_dataset(split)
        if split == 'downstream':
            dataset_grouped_by_label = [dataset.filter(lambda example: example['label'] == i) for i in range(len(self.labellist))]
            for i in range(len(self.labellist)):
                assert dataset_grouped_by_label[i].num_rows >= 2 * k_shot
                dataset_grouped_by_label[i] = dataset_grouped_by_label[i].shuffle(seed=seed).select(list(range(2 * k_shot))).train_test_split(test_size=k_shot, shuffle=False)
            train_dataset = concatenate_datasets([ds['train'] for ds in dataset_grouped_by_label])
            dev_dataset = concatenate_datasets([ds['test'] for ds in dataset_grouped_by_label])
            dataset = DatasetDict(train=train_dataset, dev=dev_dataset)
        return dataset


class MultipleChoiceQABasicDataset(BasicDataset):
    def __init__(self, path, n_prompt_tokens, has_test):
        super(MultipleChoiceQABasicDataset, self).__init__(path, has_test, n_prompt_tokens)
        self.max_len = 510 - n_prompt_tokens

    def convert_examples(self, example):
        input_ids = [101] + self.init_prompt + self.tokenizer.encode(self.input_template(example))[1:-1]
        options = [self.tokenizer.encode(opt)[1:-1] for opt in example['options']]
        option_len = [len(opt) for opt in options]
        label_mask = []
        option_seq = []
        for i in range(len(options)):
            option_seq.extend(options[i] + [8024])
            label_mask.extend([1] + [0] * (option_len[i] - 1) + [1])
        input_ids = input_ids[:self.max_len - len(option_seq)]
        input_ids_len = len(input_ids)
        label = example['label']
        start_positions = input_ids_len + sum(option_len[:label]) + label
        return {
            'input_ids': input_ids + option_seq[:-1] + [102],
            'start_positions': start_positions,
            'end_positions': start_positions + len(options[label]) - 1,
            'label_mask': [0] * input_ids_len + label_mask,
            'label': label
        }

    def get_dataset(self, split='train', k_shot=32, seed=42):
        dataset = self._get_dataset(split)
        if split == 'downstream':
            dataset = dataset.shuffle(seed=seed)
            assert dataset.num_rows >= 2 * k_shot
            dataset = dataset.select(list(range(2 * k_shot))).train_test_split(test_size=k_shot, shuffle=False)
            dataset['dev'] = dataset.pop('test')
        return dataset


class ExtractiveQABasicDataset(BasicDataset):
    def __init__(self, path, n_prompt_tokens, has_test, has_is_impossible):
        super(ExtractiveQABasicDataset, self).__init__(path, has_test, n_prompt_tokens)
        if has_is_impossible:
            self.hard_prompt = self.tokenizer.encode('抽取式问答（可能没有答案）：文本"')[1:-1]
        else:
            self.hard_prompt = self.tokenizer.encode('抽取式问答：文本"')[1:-1]
        self.hard_prompt_len = len(self.hard_prompt)
        self.has_is_impossible = has_is_impossible

    def get_dataset(self, split='train', k_shot=32, seed=42):
        dataset = self._get_dataset(split).filter(lambda x: len(x['input_ids']) <= 512)
        if split == 'downstream':
            dataset = dataset.shuffle(seed=seed)
            assert dataset.num_rows >= 2 * k_shot
            dataset = dataset.select(list(range(2 * k_shot))).train_test_split(test_size=k_shot, shuffle=False)
            dataset['dev'] = dataset.pop('test')
        return dataset

    def convert_examples(self, example):
        context_ids = self.tokenizer(example['context'])
        if self.has_is_impossible and example['is_impossible']:
            start_positions = 9 + self.n_prompt_tokens
            end_positions = start_positions + 3
        else:
            start_positions = context_ids.char_to_token(
                example['answer_start']) + self.n_prompt_tokens + self.hard_prompt_len
            end_positions = context_ids.char_to_token(
                example['answer_start'] + len(example['answer_text']) - 1) + self.n_prompt_tokens + self.hard_prompt_len
        input_ids = [101] + self.init_prompt + self.hard_prompt + context_ids['input_ids'][1:-1] + self.tokenizer.encode(
            f'问题"{example["question"]}')[1:]
        return {
            'input_ids': input_ids,
            'start_positions': start_positions,
            'end_positions': end_positions
        }


class AFQMCDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/AFQMC/AFQMC.py',
            labellist=["不同", "相似"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'意思判别："{example["text1"]}"与"{example["text2"]}"的意思是？选项：'


class OcnliDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/ocnli/ocnli.py',
            labellist=["矛盾", "中立", "蕴含"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'意思判别："{example["text1"]}"与"{example["text2"]}"的关系是？选项：'


class PawsDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/paws/paws.py',
            labellist=["矛盾", "中立", "蕴含"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'意思判别："{example["text1"]}"与"{example["text2"]}"的关系是？选项：'


class CMNLIDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/CMNLI/cmnli.py',
            labellist=["矛盾", "中立", "蕴含"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'意思判别："{example["text1"]}"与"{example["text2"]}"的关系是？选项：'


class ChnSentiCorpDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/chnsenticorp/chnsenticorp.py',
            labellist=["负面", "正面"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'情感分析："{example["text"]}"的情感是？选项：'


class THUCNewsDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/THUCNews/thuc_news.py',
            labellist=["体育", "娱乐", "财经", "教育", "时尚", "八卦", "游戏", "社会", "科技", "经济"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1] * 9 + [1, 0]
        )

    def input_template(self, example):
        return f'主题识别："{example["text"]}"的主题是？选项：'


class BQDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(BQDataset, self).__init__(
            path=f'{self.data_dir}/bq_corpus/bq_corpus.py',
            labellist=["不同", "相同"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'意思判别："{example["text1"]}"与"{example["text2"]}"的意思是？选项：'


class ChipCtcDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(ChipCtcDataset, self).__init__(
            path=f'{self.data_dir}/CHIP_CTC/CHIP_CTC.py',
            labellist=['疾病', '症状', '迹象', '孕期', '肿瘤', '过敏', '口腔', '药学', '疗法', '设备', '护理', '诊断', '年龄',
                       '性别', '教育', '地址', '种族', '意愿', '容量', '伦理', '睡眠', '运动', '饮食', '吸烟', '献血', '就医',
                       '残障', '健康', '数据', '综合', '饮酒者', '性相关', '符合协议', '成瘾行为', '器官组织', '预期寿命',
                       '风险评估', '受体状态', '病情发展', '特殊体征', '专业知识', '实验室检查', '研究者决策', '参与其他研究'],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1] * 30 + [1, 0, 0, 1] * 2 + [1, 0, 0, 0, 1] * 9 + [1, 0, 0, 0, 0, 1] * 2 + [1, 0, 0, 0, 0, 0]
        )

    def input_template(self, example):
        return f'主题识别："{example["text"]}"的主题是？选项：'


class ChipStsDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(ChipStsDataset, self).__init__(
            path=f'{self.data_dir}/CHIP_STS/CHIP_STS.py',
            labellist=["不同", "相似"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'意思判别："{example["text1"]}"与"{example["text2"]}"的意思是？选项：'


class ClueWSCDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(ClueWSCDataset, self).__init__(
            path=f'{self.data_dir}/cluewsc/cluewsc.py',
            labellist=["不同", "相同"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'指代判别:在句子"{example["text"]}"中，代词"{example["pronoun"]}"指代的和"{example["quote"]}"一致吗？选项：'


class CSLDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(CSLDataset, self).__init__(
            path=f'{self.data_dir}/csl/csl.py',
            labellist=["不同", "相同"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'主题判别:"{example["abst"]}"和关键词"{example["keyword"]}"一致吗？选项：'


class FinReDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(FinReDataset, self).__init__(
            path=f'{self.data_dir}/fin_re/fin_re.py',
            labellist=['未知', '注资', '拥有', '纠纷', '自己', '增持', '重组', '买资', '签约', '持股', '交易', '入股', '转让', '成立', '分析', '合作',
                       '帮助', '发行', '商讨', '合并', '竞争', '订单', '减持', '合资', '收购', '借壳', '欠款', '被发行', '被转让', '被成立', '被注资',
                       '被持股', '被拥有', '被收购', '被帮助', '被借壳', '被买资', '被欠款', '被增持', '拟收购', '被减持', '被分析', '被入股', '被拟收购'],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1] * 27 + [1, 0, 0, 1] * 16 + [1, 0, 0, 0]
        )

    def input_template(self, example):
        return f'关系判别：主语"{example["subject"]}"和宾语"{example["object"]}"在句子"{example["text"]}"中的关系是？选项：'


class C3Dataset(MultipleChoiceQABasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(C3Dataset, self).__init__(
            path=f'{self.data_dir}/C3/C3.py',
            n_prompt_tokens=n_prompt_tokens,
            has_test=False
        )

    def input_template(self, example):
        return f'阅读选择：文档"{example["document"]}"，问题"{example["question"]}",选项：'


class DogWhistleDataset(MultipleChoiceQABasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(DogWhistleDataset, self).__init__(
            path=f'{self.data_dir}/Dog_whistle/dog_whistle.py',
            n_prompt_tokens=n_prompt_tokens,
            has_test=False
        )

    def input_template(self, example):
        return f'近义词选择：与词语"{example["question"]}"最相近的词是？选项：'


# class CAILDataset(ExtractiveQABasicDataset):
#     def __init__(self, n_prompt_tokens=50):
#         super(CAILDataset, self).__init__(
#             path=f'{self.data_dir}/CAIL/CAIL.py',
#             n_prompt_tokens=n_prompt_tokens,
#             has_test=False,
#             has_is_impossible=True
#         )


class Cmrc2018Dataset(ExtractiveQABasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(Cmrc2018Dataset, self).__init__(
            path=f'{self.data_dir}/cmrc2018/cmrc2018.py',
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            has_is_impossible=False
        )


class DRCDDataset(ExtractiveQABasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(DRCDDataset, self).__init__(
            path=f'{self.data_dir}/drcd/drcd.py',
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            has_is_impossible=False
        )


class DuReaderChecklistDataset(ExtractiveQABasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(DuReaderChecklistDataset, self).__init__(
            path=f'{self.data_dir}/dureader_checklist/dureader_checklist.py',
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            has_is_impossible=True
        )


class DuReaderRobustDataset(ExtractiveQABasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(DuReaderRobustDataset, self).__init__(
            path=f'{self.data_dir}/dureader_robust/dureader_robust.py',
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            has_is_impossible=False
        )


class Fudan_tcDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/Fudan_tc/fudan_tc.py',
            labellist=["艺术", "文学", "教育", "哲学", "历史", "空间", "能源", "电力", "交流",
                       "计算机", "矿业", "运输", "环境", "建筑", "金融", "法律", "医药", "军事", "政治", "体育"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1] * 9 + [1, 0, 0, 1] + [1, 0, 1] * 9 + [1, 0]
        )

    def input_template(self, example):
        return f'主题识别："{example["text"]}"的主题是？选项：'


# class iflytekDataset(TCNLIBasicDataset):
#     def __init__(self, n_prompt_tokens=50):
#         super().__init__(
#             path=f'{self.data_dir}/iflytek/iflytek.py',
#             labellist=["打车", "地图导航", "免费WIFI", "租车", "同城服务", "快递物流", "婚庆", "家政", "公共交通", "政务", "社区服务", "薅羊毛", "魔幻",
#                        "仙侠", "卡牌", "飞行空战", "射击游戏", "休闲益智", "动作类", "体育竞技", "棋牌中心", "经营养成", "策略", "MOBA", "辅助工具", "约会社交",
#                        "即时通讯", "工作社交", "论坛圈子", "婚恋社交", "情侣社交", "社交工具", "生活社交", "微博博客", "新闻", "漫画", "小说", "技术", "教辅",
#                        "问答交流", "搞笑", "杂志", "百科", "影视娱乐", "求职", "兼职", "视频", "短视频", "音乐", "直播", "电台", "K歌", "成人", "中小学",
#                        "职考", "公务员", "英语", "视频教育", "高等教育", "成人教育", "艺术", "语言(非英语)", "旅游资讯", "综合预定", "民航", "铁路", "酒店",
#                        "行程管理", "民宿短租", "出国", "工具", "亲子儿童", "母婴", "驾校", "违章", "汽车咨询", "汽车交易", "日常养车", "行车辅助", "租房", "买房",
#                        "装修家居", "电子产品", "问诊挂号", "养生保健", "医疗服务", "减肥瘦身", "美妆美业", "菜谱", "餐饮店", "体育咨讯", "运动健身", "支付", "保险",
#                        "股票", "借贷", "理财", "彩票", "记账", "银行", "美颜", "影像剪辑", "摄影修图", "相机", "绘画", "二手", "电商", "团购", "外卖",
#                        "电影票务", "社区超市", "购物咨询", "笔记", "办公", "日程管理", "女性", "经营", "收款", "其他"],
#             n_prompt_tokens=n_prompt_tokens,
#             has_test=True,
#             label_mask=[]
#         )
#
#     def input_template(self, example):
#         return f'主题识别："{example["text"]}"的主题是？选项：'


class KUAKE_QICDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/KUAKE_QIC/KUAKE_QIC.py',
            labellist=["治疗方案", "疾病表述", "指标解读", "病情诊断", "就医建议", "注意事项", "后果表述", "病因分析", "功效作用", "医疗费用", "其他"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 0, 0, 1] * 10 + [1, 0]
        )

    def input_template(self, example):
        return f'主题识别："{example["text"]}"的主题是？选项：'


# 同下
class nlpcc_dbqaDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/nlpcc_dbqa/nlpcc_dbqa.py',
            labellist=["矛盾", "蕴含"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'意思判别："{example["text1"]}"与"{example["text2"]}"的关系是？选项：'


# 这个任务不知道012指代什么
class KUAKE_QQRDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/KUAKE_QQR/KUAKE_QQR.py',
            labellist=["矛盾", "中立", "蕴含"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'意思判别："{example["text1"]}"与"{example["text2"]}"的关系是？选项：'
    # 同上


# class KUAKE_QTRDataset(TCNLIBasicDataset):
#     def __init__(self, n_prompt_tokens=50):
#         super().__init__(
#             path=f'{self.data_dir}/KUAKE_QTR/KUAKE_QTR.py',
#             labellist=['0', '1', '2', '3'],
#             n_prompt_tokens=n_prompt_tokens,
#             has_test=True
#         )
#
#     def input_template(self, example):
#         return f'意思判别："{example["text1"]}"与"{example["text2"]}"的关系是？选项：'

class LCQMCDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/LCQMC/LCQMC.py',
            labellist=["匹配", "不符"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'意思判别："{example["text1"]}"与"{example["text2"]}"的关系是？选项：'


# # 这个感觉读入的时候有所不同，他是读入T or F的
# class nlpcc_emotion_tcDataset(TCNLIBasicDataset):
#     def __init__(self, n_prompt_tokens=50):
#         super().__init__(
#             path=f'{self.data_dir}/nlpcc_emotion_tc/nlpcc_emotion_tc.py',
#             labellist=["快乐", "悲伤", "愤怒", "恐惧", "惊喜"],
#             n_prompt_tokens=n_prompt_tokens,
#             has_test=True
#         )
#
#     def input_template(self, example):
#         return f'主题识别："{example["text"]}"的主题是？选项：'

class nlpcc_tcDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/nlpcc_tc/nlpcc_tc.py',
            labellist=["负面", "正面"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'情感识别："{example["text"]}"的主题是？选项：'


class SanWenDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/SanWen/sanwen.py',
            labellist=["未知", "创建", "使用", "贴近", "位于", "占有", "社会相关", "家庭关系", "一般与特别", "部分与整体"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1] * 6 + [1, 0, 0, 0, 1] * 2 + [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
        )

    def input_template(self, example):
        return f'关系判别：主语"{example["subject"]}"和宾语"{example["object"]}"在句子"{example["text"]}"中的关系是？选项：'


class tnewsDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/tnews/tnews.py',
            labellist=["房产", "汽车", "金融", "体育", "文化", "娱乐", "教育", "科技", "军事", "旅游", "世界", "农业", "股票",
                       "游戏", "故事"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1] * 14 + [1, 0]
        )

    def input_template(self, example):
        return f'主题识别："{example["sentence"]}"的主题是？选项：'


class toutiao_tcDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/toutiao_tc/toutiao_tc.py',
            labellist=["房产", "汽车", "金融", "体育", "文化", "娱乐", "教育", "科技", "军事", "旅游", "世界", "农业", "股票",
                       "游戏", "故事"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1] * 14 + [1, 0]
        )

    def input_template(self, example):
        return f'主题识别："{example["text"]}"的主题是？选项：'


class xnliDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/xnli/xnli_zh.py',
            labellist=["矛盾", "中立", "蕴含"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'意思判别："{example["text1"]}"与"{example["text2"]}"的关系是？选项：'


class CoteBdDataset(ExtractiveQABasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(CoteBdDataset, self).__init__(
            path=f'{self.data_dir}/COTE_BD/cote_bd.py',
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            has_is_impossible=False
        )


class CoteDpDataset(ExtractiveQABasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(CoteDpDataset, self).__init__(
            path=f'{self.data_dir}/COTE_DP/cote_dp.py',
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            has_is_impossible=False
        )


class CoteMfwDataset(ExtractiveQABasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(CoteMfwDataset, self).__init__(
            path=f'{self.data_dir}/COTE_MFW/cote_mfw.py',
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            has_is_impossible=False
        )


class AmazonDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/amazon/amazon.py',
            labellist=["非常差", "较差", '一般', '较好', '非常好'],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 0, 1] + [1, 0, 1] * 3 + [1, 0, 0]
        )

    def input_template(self, example):
        return f'打分："{example["text"]}"的评价是？选项：'


class BaoxianzhidaoDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/baoxianzhidao/baoxianzhidao.py',
            labellist=["符合", "不符"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'问答判断："{example["question"]}"的回答"{example["reply"]}"符合问题吗？选项：'


class DianxinzhidaoDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/dianxinzhidao/dianxinzhidao.py',
            labellist=["符合", "不符"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'问答判断："{example["question"]}"的回答"{example["reply"]}"符合问题吗？选项：'


class FinancezhidaoDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/financezhidao/financezhidao.py',
            labellist=["符合", "不符"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'问答判断："{example["question"]}"的回答"{example["reply"]}"符合问题吗？选项：'


class LawzhidaoDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/lawzhidao/lawzhidao.py',
            labellist=["符合", "不符"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'问答判断："{example["question"]}"的回答"{example["reply"]}"符合问题吗？选项：'


class LiantongzhidaoDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/liantongzhidao/liantongzhidao.py',
            labellist=["符合", "不符"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'问答判断："{example["question"]}"的回答"{example["reply"]}"符合问题吗？选项：'


class NonghangzhidaoDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/nonghangzhidao/nonghangzhidao.py',
            labellist=["符合", "不符"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'问答判断："{example["question"]}"的回答"{example["reply"]}"符合问题吗？选项：'


class TouzizhidaoDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/touzizhidao/touzizhidao.py',
            labellist=["符合", "不符"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'问答判断："{example["question"]}"的回答"{example["reply"]}"符合问题吗？选项：'


class CCPMDataset(MultipleChoiceQABasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(CCPMDataset, self).__init__(
            path=f'{self.data_dir}/CCPM/ccpm.py',
            n_prompt_tokens=n_prompt_tokens,
            has_test=False
        )

    def input_template(self, example):
        return f'诗句理解：与句子"{example["document"]}"最相近的诗句是？选项：'


class DianpingDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/dianping/dianping.py',
            labellist=["非常差", "较差", '一般', '较好', '非常好'],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 0, 1] + [1, 0, 1] * 3 + [1, 0, 0]
        )

    def input_template(self, example):
        return f'打分："{example["text"]}"的评价是？选项：'


class DMSCDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/dmsc/dmsc.py',
            labellist=["非常差", "较差", '一般', '较好', '非常好'],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 0, 1] + [1, 0, 1] * 3 + [1, 0, 0]
        )

    def input_template(self, example):
        return f'打分："{example["text"]}"的评价是？选项：'


class OnlineShppingDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/online_shopping/online_shopping.py',
            labellist=["负面", "正面"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'情感分析："{example["text"]}"的情感是？选项：'


class WaimaiDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/waimai/waimai.py',
            labellist=["负面", "正面"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'情感分析："{example["text"]}"的情感是？选项：'


class WeiboSentimentDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path=f'{self.data_dir}/weibo_sentiment/weibo_sentiment.py',
            labellist=["负面", "正面"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            label_mask=[1, 0, 1, 1, 0]
        )

    def input_template(self, example):
        return f'情感分析："{example["text"]}"的情感是？选项：'


Dataset_list = [
    AFQMCDataset,
    # OcnliDataset,
    PawsDataset,
    CMNLIDataset,
    # ChnSentiCorpDataset,
    CSLDataset,
    THUCNewsDataset,
    BQDataset,
    ChipCtcDataset,
    # DRCDDataset,
    DogWhistleDataset,
    FinReDataset,
    DuReaderChecklistDataset,
    DuReaderRobustDataset,
    ChipStsDataset,
    # C3Dataset,
    Cmrc2018Dataset,
    ClueWSCDataset,
    Fudan_tcDataset,
    KUAKE_QICDataset,
    KUAKE_QQRDataset,
    # LCQMCDataset,
    nlpcc_tcDataset,
    SanWenDataset,
    # tnewsDataset,
    toutiao_tcDataset,
    xnliDataset,
    nlpcc_dbqaDataset,
    # CoteBdDataset,
    CoteDpDataset,
    CoteMfwDataset,
    CCPMDataset,
    AmazonDataset,
    BaoxianzhidaoDataset,
    DianpingDataset,
    DianxinzhidaoDataset,
    DMSCDataset,
    FinancezhidaoDataset,
    LawzhidaoDataset,
    LiantongzhidaoDataset,
    NonghangzhidaoDataset,
    OnlineShppingDataset,
    TouzizhidaoDataset,
    WaimaiDataset,
    WeiboSentimentDataset
]

num_datasets = len(Dataset_list)

if __name__ == '__main__':
    it = get_infinite_train_iterator(32, 50)
    while True:
        next(it)


