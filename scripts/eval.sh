#!/usr/bin/env bash

gpus=2,3

data_name=LEVIR
net_G=base_transformer_pos_s4_dd8_dedim8
#base_transformer_pos_s4_dd8
#base_resnet18
#base_transformer_pos_s4_dd8
#
split=test
project_name=BIT_LEVIR
checkpoint_name=best_ckpt.pt

python eval_cd.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


