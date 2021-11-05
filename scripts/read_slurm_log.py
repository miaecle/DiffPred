#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 17:48:24 2021

@author: zqwu
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_score(seg):
    conf_mat_lines = seg[1:5]
    conf_mat = [l.strip().strip('[').strip(']').split() for l in conf_mat_lines]
    conf_mat = np.array(conf_mat).astype(float)
    
    acc = [l for l in seg if l.startswith('Accuracy')][0]
    acc = float(acc.split()[-1])
    
    err_high = [l for l in seg if l.startswith('Error (Higher pred)')][0]
    err_high = float(err_high.split()[-1])
    
    err_low = [l for l in seg if l.startswith('Error (Lower pred)')][0]
    err_low = float(err_low.split()[-1])
    
    return conf_mat, acc, err_high, err_low
    
def read_log_to_df(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Separate into scores per epoch
    segments = []
    segment = []
    for line in lines:
        if line.startswith('Epoch'):
            segments.append(segment)
            segment = []
        segment.append(line)
    
    df = {'epoch name': [],
          's-conf mat': [],
          's-acc': [],
          's-err high': [],
          's-err low': [],
          'c-conf mat': [],
          'c-acc': [],
          'c-err high': [],
          'c-err low': [],}
    segments = [s for s in segments if s[0].startswith('Epoch')]
    for seg in segments:
        df['epoch name'].append(seg[0].split()[1])
        
        sscore_line = seg.index('Segmentation\n')
        cscore_line = seg.index('Classification\n')
        sseg = seg[sscore_line:cscore_line]
        cseg = seg[cscore_line:]
        
        sscore = extract_score(sseg)
        cscore = extract_score(cseg)
        
        df['s-conf mat'].append(sscore[0])
        df['s-acc'].append(sscore[1])
        df['s-err high'].append(sscore[2])
        df['s-err low'].append(sscore[3])
        
        df['c-conf mat'].append(cscore[0])
        df['c-acc'].append(cscore[1])
        df['c-err high'].append(cscore[2])
        df['c-err low'].append(cscore[3])        
        
    return pd.DataFrame(df)


def plot_scores(logs):
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.plot(logs['s-acc'])
    plt.title('segmentation-acc')
    
    plt.subplot(2, 2, 2)
    plt.plot(logs['s-err high'], '--')
    plt.plot(logs['s-err low'], '--')
    plt.plot(logs['s-err high'] + logs['s-err low'])
    plt.title('segmentation-error')
    
    plt.subplot(2, 2, 3)
    plt.plot(logs['c-acc'])
    plt.title('classification-acc')
    
    plt.subplot(2, 2, 4)
    plt.plot(logs['c-err high'], '--')
    plt.plot(logs['c-err low'], '--')
    plt.plot(logs['c-err high'] + logs['c-err low'])
    plt.title('classification-error')
    
    plt.tight_layout()
    return

###############################################################################
