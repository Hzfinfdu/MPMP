# Multi-Task Pre-Training of Modular Prompt for Few-Shot Learning

Implementation of Multi-Task Pre-Training of Modular Prompt for Few-Shot Learning([https://arxiv.org/abs/2210.07565](https://arxiv.org/abs/2210.07565)).

## Release
**Oct 22, 2022**. Released the first version of MPMP and the 45 datasets on [Google Drive](https://drive.google.com/file/d/1wPzPR0fsD7PYssdb4TyFKFSMrxiHF2MR/view?usp=sharing).

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
Please modify the meta path to the directory that contains the datasets in `dataconfig.py`.

Use the default parameters in run.py to reproduce the training of MPMP.
```bash
sh run.sh
```