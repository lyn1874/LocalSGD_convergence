#!/bin/bash
trap "exit" INT
lr=${1?:Error: the initial learning rate}
lr_decay=${2?:Error: the decay schedule, cosine or flat lr}
epochs=${3?:Error: the total number of epochs}
bs=${4?:Error: the batch size}
num_device=${5?:Error: the number of devices}
num_communicate=${6?:Error: the number of communication rounds}
iid=${7?:Error: whether iid distributed or not}
version=${8?:Error: the repeating times}
alpha=${9?:Error: what is the unbalanced alpha value?}
gpu_index=${10?:Error: gpu index}

if [ "$num_device" == 1 ];then 
    python3 train.py --enc_lr "$lr" --batch_size "$bs" --epochs "$epochs" --num_device 1 --lr_decay_schedule "$lr_decay"
else
    for s_device in $num_device
    do
        for s_epoch in $epochs
        do
            python3 train_decentralised.py --enc_lr "$lr" --batch_size "$bs" --epochs "$s_epoch" --num_device "$s_device" \
                --lr_decay_schedule "$lr_decay" --num_communication "$num_communicate" --iid "$iid" --version "$version" \
                --loc nobackup --alpha "$alpha" --gpu_index "$gpu_index"
        done
    done
fi