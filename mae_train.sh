#!/bin/bash

python3 submitit_pretrain.py --nodes 1 --ngpus 2 --batch_size 256 --model deittiny --norm_pix_loss --mask_ratio 0.75 --epochs 200 --warmup_epochs 10 --blr 1.5e-4 --weight_decay 0.05
