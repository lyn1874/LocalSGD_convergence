#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   prepare_data.py
@Time    :   2022/06/18 15:36:39
@Author  :   Bo 
'''
import numpy as np 
import os 
import pickle 
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.data import random_split
import torch


class GetData(object):
    def __init__(self, num_workers, alpha=10.0):
        self.num_workers = num_workers
        mnist = np.load("mnist.npz", allow_pickle=True)
        shape = pickle.load(open("shape.obj", "rb"))
        mnist_dict = {}
        for item in mnist:
            if item == "X_train":
                mnist_dict["train"] = mnist[item]
            elif item == "X_valid":
                mnist_dict["val"] = mnist[item]
            elif item == "X_test":
                mnist_dict["test"] = mnist[item]
        self.shape = shape 
        self.mnist = mnist_dict
        self.alpha = alpha 
        
    def split_to_workers(self, seed, iid=True):
        np.random.seed(seed)
        num_shape = len(self.shape["train"])
        num_mnist = len(self.mnist["train"])
        if iid:
            num_shape_split = np.cumsum([num_shape // self.num_workers for i in range(self.num_workers - 1)] + [num_shape - num_shape // self.num_workers * (self.num_workers - 1)])
            num_mnist_split = np.cumsum([num_mnist // self.num_workers for i in range(self.num_workers - 1)] + [num_mnist - num_mnist // self.num_workers * (self.num_workers - 1)])
            shape_split = np.split(np.random.choice(np.arange(num_shape), num_shape, replace=False), 
                                   num_shape_split)[:-1]
            mnist_split = np.split(np.random.choice(np.arange(num_mnist), num_mnist, replace=False),
                                   num_mnist_split)[:-1]
        else:
            shape_split, mnist_split = [], []
            s_start, m_start = 0, 0
            shape_index = np.random.choice(np.arange(num_shape), num_shape, replace=False)
            mnist_index = np.random.choice(np.arange(num_mnist), num_mnist, replace=False)
            num_data_per_worker = (num_shape + num_mnist) // self.num_workers
            ratio_g = np.zeros([self.num_workers, 2])
            for i in range(self.num_workers):
                ratio = np.random.dirichlet([self.alpha for _ in range(2)])
                ratio_g[i] = ratio 
            ratio_g = ratio_g / np.sum(ratio_g, axis=0, keepdims=True)
            for i in range(self.num_workers):
                ratio = ratio_g[i]                
                s_end, m_end = s_start + int(num_shape * ratio[0]), m_start + int(num_mnist * ratio[1])
                shape_split.append(shape_index[s_start:s_end])
                mnist_split.append(mnist_index[m_start:m_end])
                s_start = s_end 
                m_start = m_end
            print([len(v) for v in shape_split])
            print([len(v) for v in mnist_split])
        split_group = [shape_split, mnist_split]
        return split_group 
    
    def split_data_to_workers(self, seed, iid):
        split_indices = self.split_to_workers(seed, iid)
        num_shape, num_mnist = len(self.shape["train"]), len(self.mnist["train"])
        self.shape["train"] = self.shape["train"][np.random.choice(np.arange(num_shape), num_shape, replace=False).astype(np.int32)]
        self.mnist["train"] = self.mnist["train"][np.random.choice(np.arange(num_mnist), num_mnist, replace=False).astype(np.int32)]
        tr_shape = [self.shape["train"][v] for v in split_indices[0]]
        tr_mnist = [self.mnist["train"][v] for v in split_indices[1]]
        tr_data = []
        tr_label = []
        ratio = []
        for v, q in zip(tr_shape, tr_mnist):
            tr_data.append(np.concatenate([v, q], axis=0))
            tr_label.append(np.concatenate([np.ones([len(v)]) * 0, np.ones([len(q)])]))
            ratio.append(len(v) / (len(q) + 1e-10))
        return tr_data, tr_label, ratio 
    
    def forward(self, seed, iid):
        if self.num_workers == 1:
            tr_data = np.concatenate([self.shape["train"], self.mnist["train"]], axis=0)
            tr_label = np.concatenate([np.zeros([len(self.shape["train"])]), 
                                       np.ones([len(self.mnist["train"])])], axis=0).astype(np.float32)
            tr_data = np.expand_dims(tr_data, axis=1).astype(np.float32)
        else:
            tr_data, tr_label, ratio = self.split_data_to_workers(seed, iid)
            tr_data = [np.expand_dims(v, axis=1).astype(np.float32) for v in tr_data]
            tr_label = [v.astype(np.float32) for v in tr_label]
        tt_data = np.concatenate([self.shape["test"], self.mnist["test"]], axis=0)
        tt_label = np.concatenate([np.zeros(len(self.shape["test"])), np.ones([len(self.mnist["test"])])], axis=0)
        print("The shape of the training data", np.shape(tr_data), np.shape(tr_label))
        print("The shape of the test data", np.shape(tt_data), np.shape(tt_label))
        return tr_data, tr_label, \
               np.expand_dims(np.array(tt_data), 1).astype(np.float32), tt_label.astype(np.float32) 
        
                
        
class ImageLoader(Dataset):
    def __init__(self, images, labels, transform):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        s_image = self.images[index]
        s_label = self.labels[index]
        if self.transform is not None:
            s_image = self.transform(s_image)
        return s_image.squeeze(1), s_label



def get_dataloader(tr_dataset, batch_size, num_workers=4):
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                           pin_memory=True, num_workers=num_workers)
    return tr_loader 


def get_test_dataloader(tt_dataset, batch_size, num_workers=4):
    tt_loader = DataLoader(tt_dataset, batch_size, shuffle=False, pin_memory=True, drop_last=True,
                           num_workers=num_workers)
    return tt_loader 
