    # -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:11:37 2021

@author: filipavaldeira
"""

import numpy as np


def BordaCount(P):
    
    # P is PCM matrix (not absolute)
    
    m = P.shape[0]
    
    fi = (1/m) * np.sum(P,axis=0)
    
    return np.asarray(fi).ravel()