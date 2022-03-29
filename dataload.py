import copy
from datasets import load_dataset, concatenate_datasets
import torch
import random
from transformers import BertTokenizerFast
from torch.nn.utils.rnn import pad_sequence

# torch.manual_seed(42)
# random.seed(42)
tokenizer = BertTokenizerFast.from_pretrained("fnlp/cpt-large")


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
        torch.utils.data.DataLoader(ds, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=BasicDataset.collate, pin_memory=True)
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


class BasicDataset:
    offset = 1000

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

    @staticmethod
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

    def get_infinite_dataloader(self, batch_size=32):
        return InfiniteDataLoader(self.get_dataset(), batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=self.collate)


class TCNLIBasicDataset(BasicDataset):
    def __init__(self, path, n_prompt_tokens, has_test, labellist):
        super(TCNLIBasicDataset, self).__init__(path, has_test, n_prompt_tokens)
        self.labellist = labellist
        self.tmp_labellist = copy.deepcopy(labellist)
        self.max_input_len = 510 - len('，'.join(labellist)) - n_prompt_tokens

    def convert_examples(self, example):
        input_ids = [102] + self.init_prompt + tokenizer.encode(self.input_template(example))[1:-1][:self.max_input_len]
        random.shuffle(self.tmp_labellist)
        options = '，'.join(self.tmp_labellist)
        start_positions = options.find(self.labellist[int(example['label'])]) + len(input_ids)
        return {
            'input_ids': input_ids + tokenizer.encode(options)[1:],
            'start_positions': start_positions,
            'end_positions': start_positions + len(self.labellist[int(example['label'])]) - 1
        }


class MultipleChoiceQABasicDataset(BasicDataset):
    def __init__(self, path, n_prompt_tokens, has_test):
        super(MultipleChoiceQABasicDataset, self).__init__(path, has_test, n_prompt_tokens)
        self.max_len = 510 - n_prompt_tokens

    def convert_examples(self, example):
        input_ids = [102] + self.init_prompt + tokenizer.encode(self.input_template(example))[1:-1]
        label = example['options'][example['label']]
        options = example['options']
        random.shuffle(options)
        options = '，'.join(options)
        start_positions = options.find(label) + len(input_ids)
        return {
            'input_ids': input_ids[:self.max_len - len(options)] + tokenizer.encode(options)[1:],
            'start_positions': start_positions,
            'end_positions': start_positions + len(label) - 1
        }


class ExtractiveQABasicDataset(BasicDataset):
    def __init__(self, path, n_prompt_tokens, has_test, has_is_impossible):
        super(ExtractiveQABasicDataset, self).__init__(path, has_test, n_prompt_tokens)
        if has_is_impossible:
            self.hard_prompt = tokenizer.encode('抽取式问答（可能没有答案）：文本"')[1:-1]
        else:
            self.hard_prompt = tokenizer.encode('抽取式问答：文本"')[1:-1]
        self.hard_prompt_len = len(self.hard_prompt)
        self.has_is_impossible = has_is_impossible

    def convert_examples(self, example):
        context_ids = tokenizer(example['context'])
        if self.has_is_impossible and example['is_impossible']:
            start_positions = 9 + self.n_prompt_tokens
            end_positions = start_positions + 4
        else:
            # if example['answer_start'] < 0:
            #     print(example['question'])
            #     print(example['answer_start'])
            #     print(example['answer_text'])
            start_positions = context_ids.char_to_token(example['answer_start']) + self.n_prompt_tokens + self.hard_prompt_len
            end_positions = context_ids.char_to_token(example['answer_start'] + len(example['answer_text']) - 1) + self.n_prompt_tokens + self.hard_prompt_len
        input_ids = [101] + self.init_prompt + self.hard_prompt + context_ids['input_ids'][1:-1] + tokenizer.encode(f'问题"{example["question"]}')[1:]
        return {
            'input_ids': input_ids,
            'start_positions': start_positions,
            'end_positions': end_positions
        }


class AFQMCDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path='/remote-home/share/ChineseData/chineseeval/AFQMC/AFQMC.py',
            labellist=["不同", "相似"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False
        )

    def input_template(self, example):
        return f'意思判别："{example["text1"]}"与"{example["text2"]}"的意思是？选项：'



class OcnliDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path='/remote-home/share/ChineseData/chineseeval/ocnli/ocnli.py',
            labellist=["矛盾", "中立", "蕴含"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False
        )

    def input_template(self, example):
        return f'意思判别："{example["text1"]}"与"{example["text2"]}"的关系是？选项：'

class PawsDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path='/remote-home/share/ChineseData/chineseeval/paws/paws.py',
            labellist=["矛盾", "中立", "蕴含"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=True
        )

    def input_template(self, example):
        return f'意思判别："{example["text1"]}"与"{example["text2"]}"的关系是？选项：'


class CMNLIDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path='/remote-home/share/ChineseData/chineseeval/CMNLI/cmnli.py',
            labellist=["矛盾", "中立", "蕴含"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False
        )

    def input_template(self, example):
        return f'意思判别："{example["text1"]}"与"{example["text2"]}"的关系是？选项：'


class ChnSentiCorpDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path='/remote-home/share/ChineseData/chineseeval/chnsenticorp/chnsenticorp.py',
            labellist=["负面", "正面"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=True
        )

    def input_template(self, example):
        return f'情感分析："{example["text"]}"的情感是？选项：'


class THUCNewsDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path='/remote-home/share/ChineseData/chineseeval/THUCNews/thuc_news.py',
            labellist=["体育", "娱乐", "财经", "教育", "时尚", "八卦", "游戏", "社会", "科技", "经济"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=True
        )

    def input_template(self, example):
        return f'主题识别："{example["text"]}"的主题是？选项：'


class BQDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(BQDataset, self).__init__(
            path='/remote-home/share/ChineseData/chineseeval/bq_corpus/bq_corpus.py',
            labellist=["矛盾", "中立", "蕴含"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False
        )

    def input_template(self, example):
        return f'意思判别："{example["text1"]}"与"{example["text2"]}"的关系是？选项：'


class ChipCtcDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(ChipCtcDataset, self).__init__(
            path='/remote-home/share/ChineseData/chineseeval/CHIP_CTC/CHIP_CTC.py',
            labellist=['疾病', '症状', '迹象', '孕期', '肿瘤', '病情发展', '过敏', '器官组织', '预期寿命', '口腔', '药学', '疗法', '设备', '护理', '诊断', '实验室检查', '风险评估', '受体状态', '年龄', '特殊体征', '专业知识', '性别', '教育', '地址', '种族', '意愿', '参与其他研究', '研究者决策', '容量', '伦理', '符合协议', '成瘾行为', '睡眠', '运动', '饮食', '饮酒者', '性相关', '吸烟', '献血', '就医', '残障', '健康', '数据', '综合'],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False
        )

    def input_template(self, example):
        return f'主题识别："{example["text"]}"的主题是？选项：'


class ChipStsDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(ChipStsDataset, self).__init__(
            path='/remote-home/share/ChineseData/chineseeval/CHIP_STS/CHIP_STS.py',
            labellist=["不同", "相似"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False
        )

    def input_template(self, example):
        return f'意思判别："{example["text1"]}"与"{example["text2"]}"的意思是？选项：'


class ClueWSCDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(ClueWSCDataset, self).__init__(
            path='/remote-home/share/ChineseData/chineseeval/cluewsc/cluewsc.py',
            labellist=["不同", "相同"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False
        )

    def input_template(self, example):
        return f'指代判别:在句子"{example["text"]}"中，代词"{example["pronoun"]}"指代的和"{example["quote"]}"一致吗？选项：'


class CSLDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(CSLDataset, self).__init__(
            path='/remote-home/share/ChineseData/chineseeval/csl/csl.py',
            labellist=["不同", "相同"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False
        )

    def input_template(self, example):
        return f'主题判别:"{example["abst"]}"和关键词"{example["keyword"]}"一致吗？选项：'


class FinReDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(FinReDataset, self).__init__(
            path='/remote-home/share/ChineseData/chineseeval/fin_re/fin_re.py',
            labellist=['未知', '注资', '拥有', '纠纷', '自己', '增持', '重组', '买资', '签约', '持股', '交易', '入股', '转让', '成立', '分析', '合作', '帮助', '发行', '商讨', '合并', '竞争', '订单', '减持', '合资', '收购', '借壳', '欠款', '被发行', '被转让', '被成立', '被注资', '被持股', '被拥有', '被收购', '被帮助', '被借壳', '被买资', '被欠款', '被增持', '拟收购', '被减持', '被分析', '被入股', '被拟收购'],
            n_prompt_tokens=n_prompt_tokens,
            has_test=True
        )

    def input_template(self, example):
        return f'关系判别：主语"{example["subject"]}"和宾语"{example["object"]}"在句子"{example["text"]}"中的关系是？选项：'


class C3Dataset(MultipleChoiceQABasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(C3Dataset, self).__init__(
            path='/remote-home/share/ChineseData/chineseeval/C3/C3.py',
            n_prompt_tokens=n_prompt_tokens,
            has_test=True
        )

    def input_template(self, example):
        return f'阅读选择：文档"{example["document"]}"，问题"{example["question"]}",选项：'


class DogWhistleDataset(MultipleChoiceQABasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(DogWhistleDataset, self).__init__(
            path='/remote-home/share/ChineseData/chineseeval/Dog_whistle/dog_whistle.py',
            n_prompt_tokens=n_prompt_tokens,
            has_test=True
        )

    def input_template(self, example):
        return f'近义词选择：与词语"{example["question"]}"最相近的词是？选项：'


# class CAILDataset(ExtractiveQABasicDataset):
#     def __init__(self, n_prompt_tokens=50):
#         super(CAILDataset, self).__init__(
#             path='/remote-home/share/ChineseData/chineseeval/CAIL/CAIL.py',
#             n_prompt_tokens=n_prompt_tokens,
#             has_test=True,
#             has_is_impossible=True
#         )


class Cmrc2018Dataset(ExtractiveQABasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(Cmrc2018Dataset, self).__init__(
            path='/remote-home/share/ChineseData/chineseeval/cmrc2018/cmrc2018.py',
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            has_is_impossible=False
        )


class DRCDDataset(ExtractiveQABasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(DRCDDataset, self).__init__(
            path='/remote-home/share/ChineseData/chineseeval/drcd/drcd.py',
            n_prompt_tokens=n_prompt_tokens,
            has_test=True,
            has_is_impossible=False
        )


class DuReaderChecklistDataset(ExtractiveQABasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(DuReaderChecklistDataset, self).__init__(
            path='/remote-home/share/ChineseData/chineseeval/dureader_checklist/dureader_checklist.py',
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            has_is_impossible=True
        )


class DuReaderRobustDataset(ExtractiveQABasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super(DuReaderRobustDataset, self).__init__(
            path='/remote-home/share/ChineseData/chineseeval/dureader_robust/dureader_robust.py',
            n_prompt_tokens=n_prompt_tokens,
            has_test=False,
            has_is_impossible=False
        )


class Fudan_tcDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path='/remote-home/share/ChineseData/chineseeval/Fudan_tc/fudan_tc.py',
            labellist=["艺术", "文学", "教育", "哲学", "历史", "空间", "能源", "电力", "交流",
                       "计算机", "矿业", "运输", "环境", "建筑", "金融", "法律", "医药", "军事", "政治", "体育"],
            n_prompt_tokens = n_prompt_tokens,
            has_test = True
        )

    def input_template(self, example):
        return f'主题识别：“{example["text"]}”的主题是？选项：'

class iflytekDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path='/remote-home/share/ChineseData/chineseeval/iflytek/iflytek.py',
            labellist=["打车", "地图导航", "免费WIFI", "租车", "同城服务", "快递物流", "婚庆", "家政", "公共交通", "政务", "社区服务", "薅羊毛", "魔幻", "仙侠", "卡牌", "飞行空战", "射击游戏", "休闲益智", "动作类", "体育竞技", "棋牌中心", "经营养成", "策略", "MOBA", "辅助工具", "约会社交", "即时通讯", "工作社交", "论坛圈子", "婚恋社交", "情侣社交", "社交工具", "生活社交", "微博博客", "新闻", "漫画", "小说", "技术", "教辅", "问答交流", "搞笑", "杂志", "百科", "影视娱乐", "求职", "兼职", "视频", "短视频", "音乐", "直播", "电台", "K歌", "成人", "中小学", "职考", "公务员", "英语", "视频教育", "高等教育", "成人教育", "艺术", "语言(非英语)", "旅游资讯", "综合预定", "民航", "铁路", "酒店", "行程管理", "民宿短租", "出国", "工具", "亲子儿童", "母婴", "驾校", "违章", "汽车咨询", "汽车交易", "日常养车", "行车辅助", "租房", "买房", "装修家居", "电子产品", "问诊挂号", "养生保健", "医疗服务", "减肥瘦身", "美妆美业", "菜谱", "餐饮店", "体育咨讯", "运动健身", "支付", "保险", "股票", "借贷", "理财", "彩票", "记账", "银行", "美颜", "影像剪辑", "摄影修图", "相机", "绘画", "二手", "电商", "团购", "外卖", "电影票务", "社区超市", "购物咨询", "笔记", "办公", "日程管理", "女性", "经营", "收款", "其他"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=True
        )

    def input_template(self, example):
        return f'主题识别：“{example["text"]}”的主题是？选项：'

class KUAKE_QICDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path='/remote-home/share/ChineseData/chineseeval/KUAKE_QIC/KUAKE_QIC.py',
            labellist=["治疗方案", "疾病表述", "指标解读", "病情诊断", "就医建议", "注意事项", "后果表述", "病因分析", "功效作用", "医疗费用", "其他"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=True
        )

    def input_template(self, example):
        return f'主题识别：“{example["text"]}”的主题是？选项：'

# 同下
class nlpcc_dbqaDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path='/remote-home/share/ChineseData/chineseeval/nlpcc_dbqa/nlpcc_dbqa.py',
            labellist=["矛盾", "蕴含"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False
        )

    def input_template(self, example):
        return f'意思判别：“{example["text1"]}”与“{example["text2"]}”的关系是？选项：'

# 这个任务不知道012指代什么
class KUAKE_QQRDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path='/remote-home/share/ChineseData/chineseeval/KUAKE_QQR/KUAKE_QQR.py',
            labellist=["矛盾", "中立", "蕴含"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=True
        )

    def input_template(self, example):
        return f'意思判别：“{example["text1"]}”与“{example["text2"]}”的关系是？选项：'
    # 同上

# class KUAKE_QTRDataset(TCNLIBasicDataset):
#     def __init__(self, n_prompt_tokens=50):
#         super().__init__(
#             path='/remote-home/share/ChineseData/chineseeval/KUAKE_QTR/KUAKE_QTR.py',
#             labellist=['0', '1', '2', '3'],
#             n_prompt_tokens=n_prompt_tokens,
#             has_test=True
#         )
#
#     def input_template(self, example):
#         return f'意思判别：“{example["text1"]}”与“{example["text2"]}”的关系是？选项：'

class LCQMCDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path='/remote-home/share/ChineseData/chineseeval/LCQMC/LCQMC.py',
            labellist=["匹配", "不符"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=True
        )

    def input_template(self, example):
        return f'意思判别：“{example["text1"]}”与“{example["text2"]}”的关系是？选项：'

# # 这个感觉读入的时候有所不同，他是读入T or F的
# class nlpcc_emotion_tcDataset(TCNLIBasicDataset):
#     def __init__(self, n_prompt_tokens=50):
#         super().__init__(
#             path='/remote-home/share/ChineseData/chineseeval/nlpcc_emotion_tc/nlpcc_emotion_tc.py',
#             labellist=["快乐", "悲伤", "愤怒", "恐惧", "惊喜"],
#             n_prompt_tokens=n_prompt_tokens,
#             has_test=True
#         )
#
#     def input_template(self, example):
#         return f'主题识别：“{example["text"]}”的主题是？选项：'

class nlpcc_tcDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path='/remote-home/share/ChineseData/chineseeval/nlpcc_tc/nlpcc_tc.py',
            labellist=["负面", "正面"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=True
        )

    def input_template(self, example):
        return f'主题识别：“{example["text"]}”的主题是？选项：'

class SanWenDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path='/remote-home/share/ChineseData/chineseeval/SanWen/sanwen.py',
            labellist=["未知", "创建", "使用", "贴近", "社会相关", "位于", "占有", "一般与特别", "家庭关系", "部分与整体"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=True
        )

    def input_template(self, example):
        return f'关系判别：主语“{example["subject"]}”和宾语“{example["object"]}”在句子“{example["text"]}”中的关系是？选项：'

class tnewsDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path='/remote-home/share/ChineseData/chineseeval/tnews/tnews.py',
            labellist=["房产", "汽车", "金融", "体育", "文化", "娱乐", "教育", "科技", "军事", "旅游", "世界", "农业", "股票",
                       "游戏", "故事"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=True
        )

    def input_template(self, example):
        return f'主题识别：“{example["sentence"]}”的主题是？选项：'

class toutiao_tcDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path='/remote-home/share/ChineseData/chineseeval/toutiao_tc/toutiao_tc.py',
            labellist=["房产", "汽车", "金融", "体育", "文化", "娱乐", "教育", "科技", "军事", "旅游", "世界", "农业", "股票",
                       "游戏", "故事"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=True
        )

    def input_template(self, example):
        return f'主题识别：“{example["text"]}”的主题是？选项：'

class xnliDataset(TCNLIBasicDataset):
    def __init__(self, n_prompt_tokens=50):
        super().__init__(
            path='/remote-home/share/ChineseData/chineseeval/xnli/xnli_zh.py',
            labellist=["矛盾", "中立", "蕴含"],
            n_prompt_tokens=n_prompt_tokens,
            has_test=False
        )

    def input_template(self, example):
        return f'意思判别：“{example["text1"]}”与“{example["text2"]}”的关系是？选项：'

Dataset_list = [
    AFQMCDataset,
    # OcnliDataset,
    # PawsDataset,
    # CMNLIDataset,
    # ChnSentiCorpDataset,
    # THUCNewsDataset,
    # # PawsDataset,
    # # BQDataset,
    # # ChipCtcDataset,
    DRCDDataset,
    # # DogWhistleDataset,
    # # CSLDataset,
    # # FinReDataset,
    DuReaderChecklistDataset,
    DuReaderRobustDataset,
    # ChipStsDataset,
    # C3Dataset,
    Cmrc2018Dataset,
    # ClueWSCDataset,
    # # Fudan_tcDataset,
    # # iflytekDataset,
    # KUAKE_QICDataset,
    # KUAKE_QQRDataset,
    # LCQMCDataset,
    # # nlpcc_emotion_tcDataset,
    # nlpcc_tcDataset,
    # SanWenDataset,
    # tnewsDataset,
    # toutiao_tcDataset,
    # xnliDataset,
    # nlpcc_dbqaDataset,
]

num_datasets = len(Dataset_list)


if __name__ == '__main__':
    for ds in Dataset_list:
        a = ds(2).get_dataset(split='validation')[0]
        print(tokenizer.decode(a['input_ids']))
        print(tokenizer.decode(a['input_ids'][a['start_positions']: a['end_positions'] + 1]))
