# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:30:05 2020

@author: MARINAMU
"""
import numpy as np
import pandas as pd
import os

np.random.seed(1337)
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from models_class import Models
from commons_functions import predict_data

class TrainTest1Split():
    def __init__(self, data, params):
        
        self.data = data
        self.params = params
        
        self.inputShape = data["X_train"][0].shape      
        self.max_y =  max(data["y_train"])
        self.min_y = min(data["y_train"])
        
        self.eval_type = 1 # 1= evaluate by record. 2= evaluate by segment/matrix
        self.model = None
        self.history = None
        self.results_train = None
        self.results_test = None
        self.fig_PredTrue = None
        self.fig_loss = None
        self.save_path  = None
        self.name_pref = None # name of the folder to save in
        self.rec_idx_test = None 
        self.rec_idx_train = None 
        self.dpi = 300
        
    # --------------------------------------------------------------------------------------------#
    # --------------------------------------------------------------------------------------------#
    def run_all(self):
        # 1. Create the model and compile:
        self.create_mdl()
        # 2. Run the training process
        print('\n-Training model...')
        self.train_mdl()
        # 3. Evaluate the test and train datasets:
        print('-Done training. Evaluating data...')
        self.evaluate_mdl()
        print('--Train: RMSE = {:.4}, R = {:.4}, p = {:.4}'.format(self.results_train["RMSE"], 
                                                                   self.results_train["R"], 
                                                                   self.results_train["p"]))
        print('--Test: RMSE = {:.4}, R = {:.4}, p = {:.4}'.format(self.results_test["RMSE"], 
                                                                  self.results_test["R"], 
                                                                  self.results_test["p"]))
        # Plot loss history and the predicted-true figure:
        self.plot_figures()
        # 5. Save the parameters of the whole process:
        print('-Done evaluating. Saving data...')
        self.save_all()
        print('-Done saving.\n')
    # --------------------------------------------------------------------------------------------#
    # --------------------------------------------------------------------------------------------#
    def create_mdl(self):
        mdl_define = Models(self.params["learn_rate"], self.inputShape,
                            self.params["model_name"], self.params["n_out"],
                            self.params["out_act"], [self.params["loss"]], 
                            [self.params["metric"]])
        mdl_define.define_mdl()
        self.model = mdl_define.model
    # --------------------------------------------------------------------------------------------# 
    # --------------------------------------------------------------------------------------------#
    def train_mdl(self): # Train the model and return the history (for loss plot)
        y_train_norm = self.data["y_train"]/ max(self.data["y_train"])
        
        X, y = shuffle(self.data["X_train"], y_train_norm) 
        
        callbacks = []
        if self.params["early_stopping"].get('eval', False):
            if self.params.get('valid_ratio')>0: # if there is a validation set
                early_stopping = EarlyStopping(monitor='val_loss',
                                               patience=self.params.get('early_stopping').get('patience', 20))  # the number of epochs with no improvement
            else:
                early_stopping = EarlyStopping(monitor='loss',
                                               patience=self.params.get('early_stopping').get('patience', 20))  
            callbacks = [early_stopping]
            ### The validation data is selected from the last samples in the x and y data provided, BEFORE shuffling
       
        self.history = self.model.fit(X, y,
                                      epochs = self.params["epochs"],
                                      batch_size = self.params["batch_size"],
                                      verbose = self.params.get("verbose", 0),  # (0 = silent, 1 = progress bar, 2 = one line per epoch).
                                      shuffle = 1, # whether to shuffle the training data before each epoch
                                      validation_split = self.params.get('valid_ratio'), # 27.07.2021
                                      callbacks = callbacks)
                
        
    # --------------------------------------------------------------------------------------------#
    # --------------------------------------------------------------------------------------------#
    def evaluate_mdl(self): # Plot true versus predicted:
        if self.eval_type == 1:
            # Evaluate train dataset:
            self.results_train,self.predictions_train = predict_data(self.model,
                                                                     self.data["X_train"], 
                                                                     self.data["y_train"],
                                                                     self.max_y, self.min_y)
            # Evaluate test dataset:
            self.results_test, self.predictions_test  = predict_data(self.model,
                                                                     self.data["X_test"],
                                                                     self.data["y_test"],
                                                                     self.max_y, self.min_y)
            
    # --------------------------------------------------------------------------------------------#
    # --------------------------------------------------------------------------------------------#
    def save_all(self):
        # Create the folder for the results:
        self.create_resuls_folder()
        # Save the model:
        self.save_model()
        # Save variables: 
        self.save_variables(self.save_path, self.data["X_train"], self.data["X_test"],
                            self.data["y_train"], self.data["y_test"],
                            self.results_test, self.history)
        
        # Write into txt file the parameters and results:
        self.save_parameters_results()
        # Save the predicted and the ture values:
        self.save_pred_true()
        # Save the loss figure:
        self.fig_loss.savefig(self.save_path / Path("Loss_fig.png"), 
                              dpi = self.dpi, bbox_inches='tight')
        # Save the pred vs true figure:
        self.fig_PredTrue.savefig(self.save_path / Path("PredictedVsTrue.png"),
                                  dpi = self.dpi, bbox_inches='tight')
    # --------------------------------------------------------------------------------------------#
    # --------------------------------------------------------------------------------------------#
    def save_model(self):
        model_json = self.model.to_json() 
        with open(self.save_path / Path('Model.json'), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.save_path / Path('Model_weights.h5'))
    # --------------------------------------------------------------------------------------------#
    # --------------------------------------------------------------------------------------------#
    def plot_figures(self):
        # Plot actual scores vs predicted scores of the test dataset:
        self.fig_PredTrue = self.plot_pred_true(
                self.predictions_test["y_pred"], 
                self.predictions_test["y_true"], 
                self.max_y,
                self.min_y)
        # Plot the loss function:
        self.fig_loss = self.plot_loss(self.history)
    # --------------------------------------------------------------------------------------------#
    # --------------------------------------------------------------------------------------------#        
    def save_parameters_results(self):
        
        f = open(self.save_path / Path("Parameters&Results.txt"),"w") 
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        f.writelines(short_model_summary)
        f.writelines('\nBatch size = {0}'.format(self.params["batch_size"])) 
        f.writelines('\nlearn rate = ' + str(self.params["learn_rate"]))
        f.writelines('\nNum epochs = ' + str(self.params["epochs"]))
        f.writelines('\nData file name: {}'.format(self.data["data_file_name"]))
        f.writelines('\n\nTest:\n')
        f.writelines('{}:{}\n'.format(k,v) for k, v in self.results_test.items())
        f.writelines('\n\nTrain:\n')
        f.writelines('{}:{}\n'.format(k,v) for k, v in self.results_train.items())
        f.close()

    # --------------------------------------------------------------------------------------------#
    # --------------------------------------------------------------------------------------------#
    @staticmethod
    def plot_loss(history):
        font = {'family' : 'Times New Roman'}#, 'size'   : 20}
        plt.figure(figsize=(8,7))
        plt.plot(history.history['loss'])
        plt.xlabel('#Epoch', fontsize=24, fontdict = font)
        plt.ylabel('Loss function', fontsize=24, fontdict = font)         
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.rc('font', **font)
        fig_loss = plt.gcf()
        return fig_loss
    
    # --------------------------------------------------------------------------------------------#
    # --------------------------------------------------------------------------------------------#
    @staticmethod
    def plot_pred_true(y_pred, y_true, max_y, min_y):  
        font = {'family' : 'Times New Roman'}

        plt.figure(figsize =(8, 7))
        plt.plot(np.squeeze(y_true), np.squeeze(y_pred),'ob')
        
        plt.xlabel("Actual score", fontsize=24, fontdict = font)
        plt.ylabel("Predicted score", fontsize=24, fontdict = font)
        # Plot the perfect linear (the wanted) line: 
        plt.plot(range(min_y,max_y+1), range(min_y,max_y+1), '--k')
        
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        plt.xlim(min_y-1, max_y+ 1) 
        plt.ylim(min_y-1, max_y+ 1)
        plt.rc('font', **font)
        fig_PredTrue = plt.gcf()
        return fig_PredTrue
    # --------------------------------------------------------------------------------------------#
    # --------------------------------------------------------------------------------------------#
    def create_resuls_folder(self): # Creates the folder where the results will be saved
        if self.name_pref == None:
            self.name_pref = 'RMSE {0:.4} R {1:.4}'.format(self.RMSE_test, self.R_test)
        # Create the folder to save in:        
        self.save_path = self.data["save_path"] / Path(self.name_pref)
        if os.path.isdir(self.save_path) == False:
            os.mkdir(self.save_path)
    # --------------------------------------------------------------------------------------------#
    # --------------------------------------------------------------------------------------------#
    def save_pred_true(self):
        # Write predicted and ture scores into a txt file:
        df=pd.DataFrame({'y_pred': np.squeeze(self.predictions_test['y_pred']),
                         'y_true': np.squeeze(self.predictions_test['y_true'])})
        df.to_csv(self.save_path / Path("Pred_true.txt"), sep='\t', na_rep="none", index=False)
    # --------------------------------------------------------------------------------------------#
    # --------------------------------------------------------------------------------------------#
    @staticmethod
    def save_variables(save_path, X_train, X_test, y_train, y_test, results_test, history):
        np.savez(save_path / Path("Variables"),
                 X_train_norm = X_train, 
                 X_test_norm = X_test,
                 y_train = y_train, 
                 y_test = y_test, 
                 results_test = results_test,
                 trainHistory = history.history)
        