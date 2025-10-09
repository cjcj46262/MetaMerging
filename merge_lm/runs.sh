#!/bin/bash
# run_merge.sh
# Bash脚本，用于多次运行 merge_gpt_glue.py 并方便修改超参数

torchrun --nproc_per_node=2 merge_gpt_glue.py --batch_size 8 --epoch 300 --inner_steps 0 --metalr 0.01 --adalr 0.01 --final_epochs 3
torchrun --nproc_per_node=2 merge_gpt_glue.py --batch_size 8 --epoch 300 --inner_steps 1 --metalr 0.01 --adalr 0.01 --final_epochs 3
torchrun --nproc_per_node=2 merge_gpt_glue.py --batch_size 8 --epoch 300 --inner_steps 5 --metalr 0.01 --adalr 0.01 --final_epochs 3
torchrun --nproc_per_node=2 merge_gpt_glue.py --batch_size 8 --epoch 300 --inner_steps 10 --metalr 0.01 --adalr 0.01 --final_epochs 3
torchrun --nproc_per_node=2 merge_gpt_glue.py --batch_size 8 --epoch 300 --inner_steps 1 --metalr 0.01 --adalr 0.1 --final_epochs 3

torchrun --nproc_per_node=2 merge_gpt_glue.py --batch_size 8 --epoch 300 --inner_steps 0 --metalr 0.01 --adalr 0.01 --final_epochs 1
torchrun --nproc_per_node=2 merge_gpt_glue.py --batch_size 8 --epoch 300 --inner_steps 1 --metalr 0.01 --adalr 0.01 --final_epochs 1
torchrun --nproc_per_node=2 merge_gpt_glue.py --batch_size 8 --epoch 300 --inner_steps 5 --metalr 0.01 --adalr 0.01 --final_epochs 1
torchrun --nproc_per_node=2 merge_gpt_glue.py --batch_size 8 --epoch 300 --inner_steps 10 --metalr 0.01 --adalr 0.01 --final_epochs 1
torchrun --nproc_per_node=2 merge_gpt_glue.py --batch_size 8 --epoch 300 --inner_steps 1 --metalr 0.01 --adalr 0.1 --final_epochs 1
