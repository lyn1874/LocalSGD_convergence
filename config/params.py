#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   params.py
@Time    :   2022/06/18 17:15:10
@Author  :   Bo 
'''
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def give_args():
    parser = argparse.ArgumentParser(description='Local_SGD')
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--enc_lr', type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--workers_load_data", type=int, default=8)
    parser.add_argument("--seed_use", type=int, default=1024)
    parser.add_argument("--strategy", type=str, default="dp")
    parser.add_argument("--iid", type=str2bool, default=True)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--num_device", type=int, default=1)
    parser.add_argument("--num_communication", type=int, default=10)
    parser.add_argument("--lr_decay_schedule", type=str, default="flat_lr")
    parser.add_argument("--loc", default="home", type=str)
    parser.add_argument("--gpu_index", default=10, type=int)
    return parser.parse_args()


