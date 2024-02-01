# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:23:36 2021

@author: marinamu
"""
from time import time

tics = []
class tic_toc():
    
    def tic():
        tics.append(time())
    
    def toc():
        if len(tics)==0:
            return None
        else:
            return time()-tics.pop()


