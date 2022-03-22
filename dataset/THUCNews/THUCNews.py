import csv
import datasets
# from datasets.tasks import TextClassification

_TRAIN_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1eOg09b1PKWP2yOix0jhhLMAGj79wvDpJ&export=download"
_DEV_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1vk6wsjDdspwTG8gVeZ6TV5-RkiV3wUyD&export=download"
_TEST_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1Ba2IIeFJMYBfOJsYyloHkHZ7lb3ADRwW&export=download"

_DESCRIPTION = '''
THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），
均为UTF-8纯文本格式。我们在原始新浪新闻分类体系的基础上，
重新整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。
'''

_CITATION = '''
@article{sun2016thuctc,
  title={Thuctc: an efficient chinese text classifier},
  author={Sun, Maosong and Li, Jingyang and Guo, Zhipeng and Yu, Zhao and Zheng, Y and Si, X and Liu, Z},
  journal={GitHub Repository},
  year={2016}
}'''

class THUCNews(datasets.GeneratorBasedBuilder):


    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "text": datasets.Value("string"),
                "label": datasets.features.ClassLabel(names=['sports', 'entertainment', 'property', 'education', 'fashion', 'affairs', 'game', 'sociology', 'technology', 'economics']),
            }),
            homepage='http://thuctc.thunlp.org/',
            citation=_CITATION,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        dev_path = dl_manager.download_and_extract(_DEV_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dev_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )
            for id_, row in enumerate(csv_reader):
                if id_ == 0:
                    continue
                label, text = row
                label = int(label)
                yield id_, {"text": text, "label": label}