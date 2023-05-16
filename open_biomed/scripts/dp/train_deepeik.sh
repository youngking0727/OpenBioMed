#!/bin/bash
# BBBP, Tox21, Toxcast, sider, clintox, muv, hiv, bace

DATASET="Tox21"
for seed in 0 1 2
do
python tasks/mol_task/dp_mvp_3.py \
--device cuda:0 \
--dataset MoleculeNet \
--dataset_path ../datasets/dp/moleculenet \
--dataset_name $DATASET \
--config_path ./configs/dp/deepeik.json \
--output_path ../ckpts/finetune_ckpts/dp \
--num_workers 1 \
--mode train \
--batch_size 128 \
--epochs 80 \
--weight_decay 1e-4 \
--lr 1e-4 \
--patience 20 \
--dropout 0.2 \
--seed $seed
--freeze
done


