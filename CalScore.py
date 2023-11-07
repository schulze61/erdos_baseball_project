# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 13:26:34 2023

@author: Weiqi Wang
"""

## Calculate the Brier Score based on the prediction from a model and the actual result
## Arguments:
## prob_predict: The predicted result as a 1D numpy array. It can be a deterministic
## result (0,1,0,1,...) or a winning probability of team A (0.5, 0.6, 0.7, ...) 
## actual: The actual result as a 1D numpy array with the same length as prob_predict.
## For example (1,0,0,1,0,...), 1: team A win, 0: team A lose.

## Return: 
## Brier score between 0 and 1. The smaller score means the prediction is more accurate.

import numpy as np

def cal_brier_score(prob_predict, actual):
    
    # Compare the size of the inputs
    if prob_predict.size != actual.size:
        raise ValueError('Prediction results should have the same size as the actual result')
    
    # Calculate Brier score
    score = sum((prob_predict - actual) * (prob_predict - actual))
    score = score/len(prob_predict)
    return score