#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   mlp.py
@Time    :   2022/06/18 16:42:31
@Author  :   Bo 
'''
import torch.nn as nn
import torch.nn.init as init


class MLP(nn.Module):
    def __init__(self, num_input_channel, num_class):
        super().__init__()
        self.num_class = num_class 
        self.num_input_channel = num_input_channel
        self.cls_layer = nn.Linear(self.num_input_channel, self.num_class)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
                if m.bias is not None:
                    init.normal_(m.bias, std=1e-6)
        
    def forward(self, x):
        feat = self.cls_layer(x)
        feat = nn.Sigmoid()(feat)
        return feat 
    

def loss(pred, label):
    return nn.BCELoss(reduction='sum')(pred, label).div(len(label))