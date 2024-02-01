# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:58:24 2021

@author: marinamu
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt

font = {'family': 'Times New Roman', 'size': 20}
plt.rc('font', **font)
from scipy import stats
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, StandardScaler



# load_yaml
# =============================================================================
def load_yaml(file_pointer):
    with open(file_pointer) as file:
        return yaml.full_load(file)
    
# =============================================================================    
def predict_data(model, X, y, max_y, min_y):
    y_pred = np.squeeze(np.clip(np.round(model.predict(X, verbose=0) * max_y).astype('int8'),
                                min_y, max_y))
    RMSE, R, p, R_spear = statisticcs_2_arrays(y_pred, y)
    CCC = ccc_metric(y, y_pred) 
    NRMSE = np.round(RMSE / (max_y - min_y), 4)  
    if NRMSE < 0:
        print("!!! NRMSE is NEGATIVE !!!\nRMSE={}, y_true={}".format(RMSE, y))
    # Create dictionary with the results:
    results = dict(RMSE=RMSE, R=R, p=p, R_spear=R_spear, NRMSE=NRMSE, CCC=CCC)
    predictions = dict(y_true=y, y_pred=y_pred)
    return results, predictions

# calculate Concordance Correlation Coefficient
# -------------------------------------------------------------------------
def ccc_metric(y_true, y_pred):
    """
                Concordance Correlation Coefficient
    Pearson's r measures linearity, while CCC measures agreement.
    """
    N = y_true.shape[0]
    epsilon = np.finfo(float).eps
    # covariance between y_true and y_pred
    s_xy = np.dot((y_true - np.mean(y_true)), (y_pred - np.mean(y_pred))) / \
           (N - 1.0 + epsilon)
    # means
    x_m = np.mean(y_true)
    y_m = np.mean(y_pred)
    # variances
    s_x_sq = np.var(y_true)
    s_y_sq = np.var(y_pred)

    # condordance correlation coefficient
    ccc = (2.0 * s_xy) / (s_x_sq + s_y_sq + (x_m - y_m) ** 2)
    return ccc

# =============================================================================
def statisticcs_2_arrays(y1, y2):
    y1, y2 = y1.astype('int16'), y2.astype('int16')  # int16 because ^2 is out of int8

    RMSE = np.round(np.sqrt(np.nanmean((y1 - y2) ** 2)), 4)  
    R, p = stats.pearsonr(np.squeeze(y1), np.squeeze(y2))  # Pearson’s correlation coefficient, Two-tailed p-value.
    R_spear, p_spear = stats.spearmanr(np.squeeze(y1), np.squeeze(y2),
                                       nan_policy='omit')
    # Replace NaN with 0: (in case that y_pred is all one constant number)
    R = np.nan_to_num(np.round(R, 4))
    R_spear = np.nan_to_num(np.round(R_spear, 4), 0)
    p = np.nan_to_num(np.round(p, 4))
    return RMSE, R, p, R_spear

def norm_data_by_mat(X, norm_method, transformer=[]):
    X = np.nan_to_num(X)      
    X_2D = X.reshape((X.shape[0]*X.shape[1]), X.shape[2])
    
    if bool(transformer): # if not empty (in testing).
        if norm_method == 'no': # no normalization
            X_norm = np.asarray([X_rec for X_rec in X])
        else:
            print('Using ready transformer')
            X_norm = np.asarray([transformer.transform(X_rec) for X_rec in X])
            
        if np.any(np.isnan(X_norm)):
            print('ERROR: NAN values in X_norm')
        return X_norm
    
    else: # empty -> calculate the transformer (in training)
        print('Calculating transformer')
        if norm_method == 'no': # no normalization.
            X_norm = X
            transformer = 'no norm'
        else:
            if norm_method== 'max':
                transformer = MaxAbsScaler().fit(X_2D)
            elif norm_method == 'robust':
                transformer = RobustScaler().fit(X_2D)
            elif norm_method == 'standard':
                transformer = StandardScaler().fit(X_2D)
            X_norm = np.asarray([transformer.transform(X_rec) for X_rec in X])
                
        if np.any(np.isnan(X_norm)):
            print('ERROR: NAN values in X_norm')
        return X_norm, transformer  
    
# =============================================================================
def plot_loss(history, title=''):
    plt.figure(figsize=(8, 7))
    plt.plot(history.history['loss'], label='Train')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('#Epoch', fontsize=24, fontdict=font)
    plt.ylabel('Loss function', fontsize=24, fontdict=font)
    plt.title(title, fontsize=24, fontdict=font)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.rc('font', **font)
    fig_loss = plt.gcf()
    plt.legend()
    return fig_loss

# =============================================================================
def plot_pred_true(y_pred, y_true, max_y, min_y, xlabel='', ylabel='', title='', fig=None):
    # font = {'family' : 'Times New Roman'}#, 'size'   : 20}
    slope, intercept, _, _, _ = stats.linregress(np.squeeze(y_pred),
                                                 np.squeeze(y_true))
    if fig is None:  # for subplot condition
        fig = plt.figure()

    plt.plot(np.squeeze(y_pred), np.squeeze(y_true), 'ob')
    plt.plot(np.squeeze(y_pred), intercept + slope * np.squeeze(y_pred),
             'r', label='fitted line')

    plt.xlabel(xlabel, fontsize=20, fontdict=font)
    plt.ylabel(ylabel, fontsize=20, fontdict=font)
    plt.title(title, fontsize=20, fontdict=font)
    # Plot the perfect linear (the wanted) line: 
    plt.plot(range(min_y, max_y + 1), range(min_y, max_y + 1), '--k')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlim(min_y - 1, max_y + 1)  # np.max(y_true)+1)
    plt.ylim(min_y - 1, max_y + 1)  # np.max(y_true)+1)
    plt.rc('font', **font)
    fig_PredTrue = plt.gcf()
    return fig_PredTrue