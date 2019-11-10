#!/usr/bin/env bash

# rmsprop不太适合使用

export CUDA_VISIBLE_DEVICES="1"

#先进行测试
CUDA_VISIBLE_DEVICES=1 nohup python3.6 main.py -a gru --epochs 300 --unknown_percentage 40 --model_size_info 1 10 --dct_coefficient_count 10 --window_size_ms 40 --window_stride_ms 40 --wanted_words 'up' --lr 0.01 --optimizer-type adam --logger &>nohup_base2.out&
CUDA_VISIBLE_DEVICES=1 nohup python3.6 main.py -a gru --admm-quant --resume "/home/dongpe/kws-gru/checkpoints/gru_GSC_acc_94.095_loushu.pt" --epochs 300 --unknown_percentage 40 --model_size_info 1 10 --dct_coefficient_count 10 --window_size_ms 40 --window_stride_ms 40 --wanted_words 'up' --lr 0.001 --optimizer-type adam --logger &>nohup_quant0.out&



