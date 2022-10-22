# Multi-Task Pre-Training of Modular Prompt for Few-Shot Learning

Implementation of Multi-Task Pre-Training of Modular Prompt for Few-Shot Learning([https://arxiv.org/abs/2210.07565](https://arxiv.org/abs/2210.07565)).

## Release
Oct 22, 2022. Released the first version of MPMP.
TODO: Release datasets on Google Drive.

## Installation
python 3.7 is recommended.
```bash
pip install transformers==4.11.3
pip install datasets
pip install cma
pip install fastNLP
pip install sklearn
pip install bayesian-optimization
pip install scipy
```

## Training
Use the default parameters in run.py to reproduce the training of MPMP.
```bash
sh run.sh
```

## Notes
Datasets are coming soon :)

Our main contribution and implementation details are included in the current public version. While we need to make some modification to our dataset loading scripts and upload them to Google Drive for reproduction. This may take a few days.

After that we will add more information and illustration on our method to this repo.
