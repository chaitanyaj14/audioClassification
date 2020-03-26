# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 01:06:28 2020

@author: Chaitanya
"""
import os

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)
        self.model_path = os.path.join('modelsMF', mode + '.model')
        self.p_path = os.path.join('picklesMF', mode + '.p')