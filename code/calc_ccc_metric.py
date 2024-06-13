# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 09:29:57 2024

@author: MARINAMU
"""

import numpy as np

class CalcCCC:
    def __init__(self, y_true, y_pred):
        self.y_ture = y_true
        self.y_pred = y_pred
    
    def calc_metric(self):
        """ Concordance Correlation Coefficient
        Pearson's r measures linearity, while CCC measures agreement.
        Imagine a scatterplot between the two measures:
            High agreement implies that the scatterplot points are close to the 45
            degrees line of perfect concordance which runs diagonally to the
            scatterplot, whereas a high Pearson's r implies that the scatterplot 
            points are close to any straight line.
        """
        N = self.y_ture.shape[0] 
        epsilon = np.finfo(float).eps
        # Covariance between y_true and y_pred
        s_xy = np.dot((self.y_ture - np.mean(self.y_ture)), (self.y_pred - np.mean(self.y_pred))) / \
               (N - 1.0)
        # Means
        x_m = np.mean(self.y_ture)
        y_m = np.mean(self.y_pred)
        # Variances
        s_x_sq = np.var(self.y_ture, ddof=1)
        s_y_sq = np.var(self.y_pred, ddof=1)

        return (2.0 * s_xy) / (s_x_sq + s_y_sq + (x_m - y_m) ** 2)