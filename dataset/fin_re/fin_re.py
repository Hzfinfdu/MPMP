import datasets
import os
import json
# from datasets.tasks import RelationExtraction

_CITATION = '''
@inproceedings{li-etal-2019-chinese,
    title = "{C}hinese Relation Extraction with Multi-Grained Information and External Linguistic Knowledge",
    author = "Li, Ziran  and
      Ding, Ning  and
      Liu, Zhiyuan  and
      Zheng, Haitao  and
      Shen, Ying",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1430",
    doi = "10.18653/v1/P19-1430",
    pages = "4377--4386",
    abstract = "Chinese relation extraction is conducted using neural networks with either character-based or word-based inputs, and most existing methods typically suffer from segmentation errors and ambiguity of polysemy. To address the issues, we propose a multi-grained lattice framework (MG lattice) for Chinese relation extraction to take advantage of multi-grained language information and external linguistic knowledge. In this framework, (1) we incorporate word-level information into character sequence inputs so that segmentation errors can be avoided. (2) We also model multiple senses of polysemous words with the help of external linguistic knowledge, so as to alleviate polysemy ambiguity. Experiments on three real-world datasets in distinct domains show consistent and significant superiority and robustness of our model, as compared with other baselines. We will release the source code of this paper in the future.",
}
'''

_DESCRIPTION = '''
'''

class FinRE(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                'text': datasets.Value('string'),
                'subject': datasets.Value('string'),
                'object': datasets.Value('string'),
                'predicate': datasets.features.ClassLabel(names=[
                    'unknown',
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
                    '被拟收购'
                ])
            }),
            citation=_CITATION,
            homepage='https://github.com/thunlp/Chinese_NRE/tree/master/data/FinRE',
            supervised_keys=None,
    
        )

    def _split_generators(self, dl_manager):
        files_to_download = {
            'train': '/remote-home/qzhu/prompt_pretrain/dataset/fin_re/train.txt',
            'dev': '/remote-home/qzhu/prompt_pretrain/dataset/fin_re/dev.txt',
            'test': '/remote-home/qzhu/prompt_pretrain/dataset/fin_re/test.txt',
        }
        downloaded_files = dl_manager.download_and_extract(files_to_download)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files['train']}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files['dev']}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files['test']}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, 'r', encoding="utf-8") as f:
            for _id, line in enumerate(f.readlines()):
                subject, object, predicate, text = line.strip().split('\t')
                yield _id, {
                    'text': text,
                    'subject': subject,
                    'object': object,
                    'predicate': predicate
                }
