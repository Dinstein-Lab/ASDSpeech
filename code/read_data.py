# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 12:30 2021

@author: marinamu
"""
import numpy as np
import pandas as pd
from scipy.io import loadmat
from datetime import datetime
import matplotlib.pyplot as plt
font = {'family': 'Cambria', 'size': 20}
plt.rc('font', **font)


class ReadData:
    
    def __init__(self, config, data_file_path, target_score):
        
        self.config = config
        self.data_file_path = data_file_path # data of the recordings with only one evaluation
        self.data = loadmat(self.data_file_path) 

        self.target_score = target_score
        self.year_th = config.get('year_th', 2015)
        self.unite_test_1st = config.get('unite_test_1st', False)
        self.unite_test = config.get('unite_test', False)
        self.plot_TF = config.get('plot_TF', False)
        self.fig_target_score_data = None
        self.i_mat = 0
        self.CKs = dict()
        self.dates = dict()
        self.feats_take = config.get('feats_take', 48)  # features to inlcude: -1 means all
        
        if "num_matrices" in self.data.keys():
            self.num_mats = np.squeeze(self.data['num_matrices'])
        else:
            self.num_mats = 1
        
    # =============================================================================================
    def run_all(self):
        print("*Create dataframes:")
        self.data_df = self.create_dataFrame(self.data)
        
        print("Remove old recordings (< {}):".format( self.year_th))
        self.remove_old_dates(year_th = self.year_th)
        
        # Take spesific/all gender:
        self.data_df = self.select_gender()
        
        # Take spesific module:
        self.data_df = self.select_module()
        
        print("*Create X and y:")
        self.create_X_y_1mat()
                  
        self.save_CKs_dates()  # 15.12.2021.
        
        # Summary: Print datas shapes:
        self.print_data_shape()
        # Plots:
        if self.plot_TF == True:
            self.visualize_dates()
            self.visualize_target_score()
    
    # =============================================================================================        
    def create_dataFrame(self, data):
        data_list = {'CK': [np.fromstring(ck, sep='_')[0].astype('int64') for ck in data['CK']],
                     'Date': [datetime.strptime(str(s),'%d-%m-%y').date() for s in data['record_date']],
                     self.target_score: np.squeeze(data[self.target_score].astype('int8')),
                     'Gender': np.squeeze(data["gender"]),
                     'Module': np.squeeze(data["module"])}
        return pd.DataFrame(data=data_list)
                
    # =============================================================================================
    def remove_old_dates(self, year_th = 2019):
        date_th = datetime.strptime(str(year_th),'%Y').date()         

        idx_delete = []
        for idx, rec_date in enumerate(self.data_df["Date"]):
            if rec_date < date_th:
                idx_delete.append(idx) 
        if idx_delete:
            self.data_df = self.data_df.drop(idx_delete, axis=0).reset_index(drop=True)
            print('Removed: {} recs from data_df'.format(len(idx_delete)))

    # =============================================================================================            
    def select_gender(self):
        gender = {0: 'boys', 1: 'girls'} # 1=girls,  0=boys
        if self.config["gender"] == "all":
            pass
        else:
            idx_take = np.array(self.data_df["Gender"] == self.config["gender"]) 
            print("***{} {} removed.***".format(self.data_df.shape[0] - sum(idx_take),
                                                gender[not self.config["gender"]]))
            self.data_df = self.data_df.loc[idx_take, :].reset_index(drop=True)
            
        return self.data_df
    
    # =============================================================================================            
    def select_module(self):
        if self.config["module"] == "all":
            pass
        else:
            idx_module_take = np.argwhere(np.isin(self.data_df["Module"], self.config["module"])).ravel()

            print("***{} of other module removed.***".format(self.data_df.shape[0] - len(idx_module_take)))
            self.data_df = self.data_df.loc[idx_module_take, :].reset_index(drop=True)

        return self.data_df
    # =============================================================================================
    def create_X_y_1mat(self):
        """
        Take 1 feature matrix out of num_mats for each recording.
        Data with 1 evaluation only: from data 1
        Data with 2 evaluations: First evaluation from data 1 and second evaluation from data 2
        """
        # X:
        self.X = self.data['features'][np.arange(start = self.i_mat, 
                                                 step = self.num_mats, 
                                                 stop = self.data['features'].shape[0])]
        self.X = np.asarray([X[0][:, :self.feats_take] for X in self.X]) # 21.12.21 3D array
        self.y = np.asarray(self.data_df[self.target_score])
    # =============================================================================================            
    def save_CKs_dates(self):
        self.CKs = self.data_df["CK"]
        self.dates = self.data_df["Date"]
    # =============================================================================================
    def visualize_dates(self): # Visualize date vs target score
        kwargs = dict(markersize = 12, 
                      markeredgewidth=3) # 14.11.21
        plt.figure(figsize=(15,6))
        plt.plot(self.data_df["Date"], self.data_df[self.target_score], '+', 
                 label='All data', **kwargs)       
        plt.xlabel('Date')
        plt.ylabel(self.target_score)
        plt.legend(loc = 'best', fontsize='x-small')
        self.fig_date_vs_score = plt.gcf()
        # plt.show()
        
    # =============================================================================================
    def visualize_target_score(self): # Visualize distribution of the target scores in histogram
        kwargs = dict(histtype = 'stepfilled', # filled with color, no seperation between the bars
                      alpha = 0.5, # transperency
                      edgecolor = 'black',
                      bins = np.unique(self.data_df[self.target_score])) # the bins in the xaxis
        
        plt.figure(figsize=(10,6))
        plt.hist(self.data_df[self.target_score], label='All data', **kwargs)
        plt.ylabel('Number of samples')
        plt.xlabel(self.target_score)
        plt.legend(loc = 'best', fontsize='x-small')
        self.fig_target_score_data = plt.gcf()
        # plt.show()
        
    # =============================================================================================    
    def print_data_shape(self):
        print('*'*40)
        print("Data shape: X = {}, y = {}".format(self.X.shape,
                                                   self.data_df[self.target_score].shape))
        print('*'*40)