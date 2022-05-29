CUDA_VISIBLE_DEVICES=6 python deepbbt.py --init_prompt_path '/home/ma-user/work/zfhe/BBTPrefixPretraining/results/PromptTokens50_BatchSize32_NPrompts8_LrRouter0.0005_LrPrompt0.0001_AnnealParams1.0;None;None/best.th' --task_name 'thucnews' --seed 13
CUDA_VISIBLE_DEVICES=6 python deepbbt.py --init_prompt_path '/home/ma-user/work/zfhe/BBTPrefixPretraining/results/PromptTokens50_BatchSize32_NPrompts8_LrRouter0.0005_LrPrompt0.0001_AnnealParams1.0;None;None/best.th' --task_name 'thucnews' --seed 100
CUDA_VISIBLE_DEVICES=6 python deepbbt.py --init_prompt_path '/home/ma-user/work/zfhe/BBTPrefixPretraining/results/PromptTokens50_BatchSize32_NPrompts8_LrRouter0.0005_LrPrompt0.0001_AnnealParams1.0;None;None/best.th' --task_name 'chnsenticorp' --seed 21
CUDA_VISIBLE_DEVICES=6 python deepbbt.py --init_prompt_path '/home/ma-user/work/zfhe/BBTPrefixPretraining/results/PromptTokens50_BatchSize32_NPrompts8_LrRouter0.0005_LrPrompt0.0001_AnnealParams1.0;None;None/best.th' --task_name 'chnsenticorp' --seed 42
CUDA_VISIBLE_DEVICES=6 python deepbbt.py --init_prompt_path '/home/ma-user/work/zfhe/BBTPrefixPretraining/results/PromptTokens50_BatchSize32_NPrompts8_LrRouter0.0005_LrPrompt0.0001_AnnealParams1.0;None;None/best.th' --task_name 'chnsenticorp' --seed 100
CUDA_VISIBLE_DEVICES=6 python deepbbt.py --init_prompt_path '/home/ma-user/work/zfhe/BBTPrefixPretraining/results/PromptTokens50_BatchSize32_NPrompts8_LrRouter0.0005_LrPrompt0.0001_AnnealParams1.0;None;None/best.th' --task_name 'ccpm' --seed 8
CUDA_VISIBLE_DEVICES=6 python deepbbt.py --init_prompt_path '/home/ma-user/work/zfhe/BBTPrefixPretraining/results/PromptTokens50_BatchSize32_NPrompts8_LrRouter0.0005_LrPrompt0.0001_AnnealParams1.0;None;None/best.th' --task_name 'ccpm' --seed 100
CUDA_VISIBLE_DEVICES=6 python deepbbt.py --init_prompt_path '/home/ma-user/work/zfhe/BBTPrefixPretraining/results/PromptTokens50_BatchSize32_NPrompts8_LrRouter0.0005_LrPrompt0.0001_AnnealParams1.0;None;None/best.th' --task_name 'ccpm' --seed 13
CUDA_VISIBLE_DEVICES=6 python deepbbt.py --init_prompt_path '/home/ma-user/work/zfhe/BBTPrefixPretraining/results/PromptTokens50_BatchSize32_NPrompts8_LrRouter0.0005_LrPrompt0.0001_AnnealParams1.0;None;None/best.th' --task_name 'amazon' --seed 100
CUDA_VISIBLE_DEVICES=6 python deepbbt.py --init_prompt_path '/home/ma-user/work/zfhe/BBTPrefixPretraining/results/PromptTokens50_BatchSize32_NPrompts8_LrRouter0.0005_LrPrompt0.0001_AnnealParams1.0;None;None/best.th' --task_name 'amzon' --seed 42
CUDA_VISIBLE_DEVICES=6 python deepbbt.py --init_prompt_path '/home/ma-user/work/zfhe/BBTPrefixPretraining/results/PromptTokens50_BatchSize32_NPrompts8_LrRouter0.0005_LrPrompt0.0001_AnnealParams1.0;None;None/best.th' --task_name 'drcd' --seed 21

for tn in 'c3' 'cmnli' 'ocnli' 'cmrc2018' ;do for sd in 8 13 21 42 100; do CUDA_VISIBLE_DEVICES=6 python deepbbt.py --init_prompt_path '/home/ma-user/work/zfhe/BBTPrefixPretraining/results/PromptTokens50_BatchSize32_NPrompts8_LrRouter0.0005_LrPrompt0.0001_AnnealParams1.0;None;None/best.th' --task_name $tn --seed $sd;done; done






