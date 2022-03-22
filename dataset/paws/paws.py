import json
import os
import datasets
# from datasets.tasks import TextMatching

_CITATION = '''
@article{yang2019paws,
  title={PAWS-X: A cross-lingual adversarial dataset for paraphrase identification},
  author={Yang, Yinfei and Zhang, Yuan and Tar, Chris and Baldridge, Jason},
  journal={arXiv preprint arXiv:1908.11828},
  year={2019}
}'''

_DESCRIPTION = '''
A new dataset of 23,659 human translated PAWS evaluation pairs in six typologically distinct languages: French, Spanish, German, Chinese, Japanese, and Korean. 
'''

class paws(datasets.GeneratorBasedBuilder):

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                'text1': datasets.Value('string'),
                'text2': datasets.Value('string'),
                'label': datasets.features.ClassLabel(names=['contradiction', 'neutral', 'entailment'])
            }),
            citation=_CITATION,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        files_to_download = {
            'train': '/remote-home/qzhu/prompt_pretrain/dataset/paws/train.csv',
            'dev': '/remote-home/qzhu/prompt_pretrain/dataset/paws/dev.csv',
            'test': '/remote-home/qzhu/prompt_pretrain/dataset/paws/test.csv',
        }
        downloaded_files = dl_manager.download_and_extract(files_to_download)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files['train']}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files['dev']}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files['test']}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()[1:]):
                text1, text2, label = line.strip().split('\t')
                yield id_, {
                    "text1": text1,
                    "text2": text2,
                    "label": int(label)
                }