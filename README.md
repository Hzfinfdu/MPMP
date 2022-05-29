# MPMP

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

## Example
dataload.py里面的data_dir要改为对应的路径

```bash
python deepbbt.py<or bbt.py> --init_prompt_path <path to your ckpt, currently not publicly available> --task_name 'thucnews' --seed 13
```