#!/bin/bash
# run_merge.sh

python main_emr_merging.py --batch_size 128 --meta_batch_size 8 --meta_batch_size 4 --epoch 3000 --inner_steps 1 --metalr 1 --adalr 0.1 --gpu 2
python main_emr_merging.py --batch_size 128 --meta_batch_size 8 --meta_batch_size 4 --epoch 3000 --inner_steps 1 --metalr 0.1 --adalr 0.1 --gpu 2
python main_emr_merging.py --batch_size 128 --meta_batch_size 8 --meta_batch_size 4 --epoch 3000 --inner_steps 1 --metalr 0.01 --adalr 0.1 --gpu 2
python main_emr_merging.py --batch_size 128 --meta_batch_size 8 --meta_batch_size 4 --epoch 3000 --inner_steps 1 --metalr 0.001 --adalr 0.1 --gpu 2
python main_emr_merging.py --batch_size 128 --meta_batch_size 8 --meta_batch_size 4 --epoch 3000 --inner_steps 1 --metalr 0.01 --adalr 1 --gpu 2
python main_emr_merging.py --batch_size 128 --meta_batch_size 8 --meta_batch_size 4 --epoch 3000 --inner_steps 1 --metalr 0.01 --adalr 0.1 --gpu 2
python main_emr_merging.py --batch_size 128 --meta_batch_size 8 --meta_batch_size 4 --epoch 3000 --inner_steps 1 --metalr 0.01 --adalr 0.01 --gpu 2
python main_emr_merging.py --batch_size 128 --meta_batch_size 8 --meta_batch_size 4 --epoch 3000 --inner_steps 1 --metalr 0.01 --adalr 0.001 --gpu 2