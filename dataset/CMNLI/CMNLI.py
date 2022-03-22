import datasets
import os
import json
# from datasets.tasks import TextMatching

_DESCRIPTION = '''
CMNLI数据由两部分组成：XNLI和MNLI。数据来自于fiction，telephone，travel，government，slate等，对原始MNLI数据和XNLI数据进行了中英文转化，保留原始训练集，
合并XNLI中的dev和MNLI中的matched作为CMNLI的dev，合并XNLI中的test和MNLI中的mismatched作为CMNLI的test，并打乱顺序。
该数据集可用于判断给定的两个句子之间属于蕴涵、中立、矛盾关系。'''


_CITATION = '''
@inproceedings {xu-etal-2020-clue,
 title = "{CLUE}: A {C}hinese Language Understanding Evaluation Benchmark",
 author = "Xu, Liang  and
    Hu, Hai and
    Zhang, Xuanwei and
    Li, Lu and
    Cao, Chenjie and
    Li, Yudong and
    Xu, Yechen and
    Sun, Kai and
    Yu, Dian and
    Yu, Cong and
    Tian, Yin and
    Dong, Qianqian and
    Liu, Weitang and
    Shi, Bo and
    Cui, Yiming and
    Li, Junyi and
    Zeng, Jun and
    Wang, Rongzhao and
    Xie, Weijian and
    Li, Yanting and
    Patterson, Yina and
    Tian, Zuoyu and
    Zhang, Yiwen and
    Zhou, He and
    Liu, Shaoweihua and
    Zhao, Zhe and
    Zhao, Qipeng and
    Yue, Cong and
    Zhang, Xinrui and
    Yang, Zhengliang and
    Richardson, Kyle and
    Lan, Zhenzhong ",
 booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
 month = dec,
 year = "2020",
 address = "Barcelona, Spain (Online)",
 publisher = "International Committee on Computational Linguistics",
 url = "https://aclanthology.org/2020.coling-main.419",
 doi = "10.18653/v1/2020.coling-main.419",
 pages = "4762--4772",
 abstract = "The advent of natural language understanding (NLU) benchmarks for English, such as GLUE and SuperGLUE allows new NLU models to be evaluated across a diverse set of tasks. These comprehensive benchmarks have facilitated a broad range of research and applications in natural language processing (NLP). The problem, however, is that most such benchmarks are limited to English, which has made it difficult to replicate many of the successes in English NLU for other languages. To help remedy this issue, we introduce the first large-scale Chinese Language Understanding Evaluation (CLUE) benchmark. CLUE is an open-ended, community-driven project that brings together 9 tasks spanning several well-established single-sentence/sentence-pair classification tasks, as well as machine reading comprehension, all on original Chinese text. To establish results on these tasks, we report scores using an exhaustive set of current state-of-the-art pre-trained Chinese models (9 in total). We also introduce a number of supplementary datasets and additional tools to help facilitate further progress on Chinese NLU. Our benchmark is released at https://www.cluebenchmarks.com",
}'''

class CMNLI(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                'text1': datasets.Value('string'),
                'text2': datasets.Value('string'),
                'label': datasets.features.ClassLabel(names=['contradiction', 'neutral', 'entailment'])
            }),
            citation=_CITATION,
            homepage='https://github.com/CLUEbenchmark/CLUE',
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        files_to_download = {
            'train': '/remote-home/qzhu/prompt_pretrain/dataset/CMNLI/train.json',
            'dev': '/remote-home/qzhu/prompt_pretrain/dataset/CMNLI/dev.json',
        }
        downloaded_files = dl_manager.download_and_extract(files_to_download)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files['train']}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files['dev']}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                line = json.loads(line)
                text1, text2, label = line['sentence1'], line['sentence2'], line['label']
                yield id_, {
                    'text1': text1,
                    'text2': text2,
                    'label': label
                }