import csv
import datasets

_TRAIN_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1uV-aDQoMI51A27OxVgJnzxqZFQqkDydZ&export=download"
_DEV_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1kI_LUYm9m0pVGDb4LHqtjwauUrIGo_rC&export=download"
_TEST_DOWNLOAD_URL = "https://drive.google.com/u/0/uc?id=1YJkSqGFeo-8ifmgVbxGMTte7Y-yxPHg7&export=download"
_CITATION = '''
@data{yfwt-wr77-20,
doi = {10.21227/yfwt-wr77},
url = {https://dx.doi.org/10.21227/yfwt-wr77},
author = {Tan, Songbo},
publisher = {IEEE Dataport},
title = {ChnSentiCorp},
year = {2020} }'''

_DESCRIPTION = '''This dataset is a large-scale Chinese hotel review data set collected by Tan Songbo.  
The corpus size is 10,000 reviews. The corpus is automatically collected and organized from Trip.com.
'''

class ChnSentiCorp(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=["负面", "正面"])
                }
            ),
            homepage='https://github.com/CLUEbenchmark/CLUE',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        train_path = '/remote-home/share/ChineseData/chineseeval/chnsenticorp/train.tsv'
        dev_path = '/remote-home/share/ChineseData/chineseeval/chnsenticorp/dev.tsv'
        test_path = '/remote-home/share/ChineseData/chineseeval/chnsenticorp/test.tsv'
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dev_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f.readlines()[1:]):
                label, text = line.strip().split('\t')
                label = int(label)
                yield idx, {
                    'text': text,
                    'label': label
                }