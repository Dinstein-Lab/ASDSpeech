# -*- coding: utf-8 -*-
"""
Created on Wed 01.12.2021: 11:50

@author: marinamu
"""
from pathlib import Path
import os

import numpy as np
import pickle
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import functools
import operator
import shutil
import matplotlib.pyplot as plt
import random

random.seed(1337)
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
# from memory_profiler import profile
from sklearn.model_selection import StratifiedKFold

from hyper_tune import HyperTune
from read_data import ReadData
from models_class import Models
from tic_toc_class import tic_toc
from find_min_max_score import FindMinMaxScore
from commons_functions import norm_data_by_mat, predict_data, plot_loss, plot_pred_true

# Get the current script's directory
SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
# Get the project root directory
PROJECTS_DIR = Path(os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir)))

class TrainTestKFold:

    def __init__(self, main_config, yaml_options):
        """
        Parameters:
        - main_config (dict): Main configuration parameters.
        - yaml_options (object): YAML configuration options.
        """
        self.main_config = main_config
        self.yaml_options = yaml_options
        self.paths_config = main_config['paths_config']
        self.params_config = main_config['params_config']
        self.tune_params_config = main_config['hyper_tune_config']
        self.norm_method = self.params_config['norm_method']
        self.data_norm_by = self.params_config['data_norm_by']
        self.model_name = self.params_config['model_name']
        self.target_score = self.params_config['score_name']
        self.data_file_path = PROJECTS_DIR / self.paths_config['data_file_path']
        self.num_mats_take = self.params_config["num_mats_take"]  # on how many matrices to run
        self.statistic = self.tune_params_config["statistic"]
        self.i_mat = self.params_config['i_mat']
        self.k_folds = self.params_config['k_folds']
        
        self.best_params = "not_defined"  # dictionary
        self.dpi = 300  # saving images resolution
        self.dataset_names = ['train', 'test']
        self.random_state = self.params_config.get('random_state', 1337)

        self.save_path = None
        self.min_y, self.max_y = FindMinMaxScore(
            target_score_name=self.params_config["score_name"]).calculate_min_max()  # self.find_min_max_target() # 03.10.23
        self.script_path = SCRIPT_DIR

    # =============================================================================================
    def run_all(self):
        """
        Run all steps of the training and testing process.

        Returns:
        None
        """
        # Load data with first feature mat for each record:
        data_class = self.load_data()

        #  Create save path:
        self.create_save_path(data_class)
        
        # Create save path of the chosen statistic (Best_R_rec_mean, Best_CCC_rec_mean)...
        self.create_save_path_statistic()

        # Copy yaml file to save path:
        self.copy_yaml_to_save_path()
        
        # Copy the script folder to the save path:  
        print('**Copying the scripts to save path...')
        self.copy_script_folder()
        
        if self.params_config.get('calculate'):
            # Train and predict using the chosen hyperparameters:
            self.train_test_kfolds()

    # =============================================================================================
    # @profile
    def load_data(self, i_mat=0):
        """
        Parameters
        ----------
        - i_mat (int): The number of random feature matrix to take from each recording.


        Returns
        -------
        data_class (object): Loaded data object.
        """

        # Create one split of Train-Test with one feature matrix for each recording:
        data_class = ReadData(config=self.params_config,
                              data_file_path=self.data_file_path,
                              target_score=self.target_score)
        data_class.i_mat = i_mat
        data_class.run_all()
        return data_class

    # =============================================================================================
    def create_save_path(self, data_class):
        """
        Create the save path for the results.

        Parameters:
        - data_class (object): Loaded data object.

        Returns:
        None
        """
        
        n_recs = data_class.data_df.shape[0]
        name_main_folder = '{}recs_{}folds'.format(n_recs, self.k_folds)

        # Create the save path: 
        self.save_path = self.create_sub_save_path(name_main_folder) / self.model_name

        if os.path.isdir(self.save_path) is False:
            os.makedirs(self.save_path)
            print('Created saving path: ', self.save_path, '\n')

    # =============================================================================================
    def create_sub_save_path(self, name_main_folder):
        """
        Create a sub save path.

        Parameters:
        - name_main_folder (str): Main folder name.

        Returns:
        sub_save_path (object): Sub save path object.
        """
        return PROJECTS_DIR / self.paths_config.get('save_path', 'results') /\
                        '{}_prediction'.format(self.params_config["score_name"].upper()) /\
                            name_main_folder
    
    # =============================================================================================
    def create_save_path_statistic(self):
        """
       Create the save path for the chosen statistic.

       Returns:
       None
       """
        self.save_path_statistic = self.save_path / rf"Best_{self.statistic}"
        
    # =============================================================================================
    def copy_yaml_to_save_path(self):
        """
        Copy YAML configuration file to the save path.

        Returns:
        None
        """
        print(f"save_path_statistic={self.save_path_statistic}")
        if not os.path.exists(self.save_path_statistic): 
            os.makedirs(self.save_path_statistic)
        # Copy the configuration file to the save path:
        shutil.copy(self.yaml_options.config, self.save_path_statistic)
        
    # =============================================================================================           
    def copy_script_folder(self, extension='.py'):  
        """
        Copy script files to the save path.

        Parameters:
        - extension (str): File extension to copy.

        Returns:
        None
        """
        destination_folder = self.save_path_statistic / Path('scripts')
        if not os.path.exists(destination_folder): 
            os.makedirs(destination_folder)

        for filename in os.listdir(self.script_path):
            if filename.endswith(extension):  # only files that ends with "extension"
                shutil.copy(self.script_path / Path(filename), destination_folder / Path(filename))

    # =============================================================================================
    def define_tune_params(self):
        # Define parameters:
        bs_config = self.tune_params_config.get('batch_size')
        ep_config = self.tune_params_config.get('n_epochs')
        lr_config = self.tune_params_config.get('learn_rate')
        # Create batch size variations based on base of 2: 2^3, 2^4,...
        self.batch_size = np.power(2, np.arange(start=bs_config.get('start'),
                                                stop=bs_config.get('stop'), dtype=int))
        # Create number of epochs variations: from start:step:stop
        self.n_epochs = np.arange(start=ep_config.get('start'), stop=ep_config.get('stop'),
                                  step=ep_config.get('step'), dtype=int)
        # Create learning rate variations based on base of 10: 10^(-3)...
        lr_range = np.power(10, np.arange(start=lr_config.get('start'),
                                          stop=lr_config.get('stop'),
                                          dtype=float))  # [1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01]
        self.learn_rate = np.sort(np.concatenate((lr_range, lr_range * 5), axis=None))[:-1]
        tune_params = dict(batch_size=self.batch_size,
                           epochs=self.n_epochs,
                           learn_rate=self.learn_rate,
                           n_out=[self.params_config.get('n_out')],
                           out_act=[self.params_config.get('act_out')],
                           loss=[self.params_config.get('loss_func')],
                           metric=[self.params_config.get('metric')],
                           norm_method=[self.params_config.get('norm_method')])
        return tune_params

    # =============================================================================================
    def hyper_tune(self, data_train, save_path):
        tic_toc.tic()
        self.tune_save_path = save_path / r"Hyper_tune"
        if not os.path.isdir(self.tune_save_path):
            os.makedirs(self.tune_save_path)
        print("Hyper-tuning save folder: {}\n".format(self.tune_save_path))
        # Define a dictionary of the parameters for tuning:
        tuning_params = dict()
        tuning_params["X"] = data_train["X"]
        tuning_params["y"] = data_train["y"]
        tuning_params["param_grid"] = self.define_tune_params()
        tuning_params["save_path"] = self.tune_save_path
        tuning_params["data_filename"] = self.data_file_path

        # Hyper tune parameters:
        if self.tune_params_config["calculate"] == 'tune' or \
                (self.tune_params_config["calculate"] == 'load' and
                 not os.path.isfile(save_path / Path(
                     'Hyper_tune/Mean_CV_results.csv'))):  # if the excel file doesnt exist
            hyper_tune_cl = HyperTune(tuning_params, self.params_config,
                                      self.tune_params_config)
            hyper_tune_cl.prepare_data_TF = False  # Do Not normalize because the X is already normalized
            hyper_tune_cl.run_all()
            self.best_params = hyper_tune_cl.best_params

        elif self.tune_params_config["calculate"] == 'read':
            print('*Loading model parameters from yaml file*')
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            self.best_params = self.tune_params_config['best_params']

        # Load the parameters from fold's folder Hyper_tune: 
        elif self.tune_params_config["calculate"] == 'load':
            print('*Loading model parameters from the Hyper_tune folder*')
            hyper_tune_cl = HyperTune(tuning_params, self.params_config,
                                      self.tune_params_config)
            hyper_tune_cl.load_best_params()  # Load the parameters from the excel file
            self.best_params = hyper_tune_cl.best_params

        print('********* Tunning took {}s *********'.format(tic_toc.toc()))

    # =============================================================================================
    def train_test_kfolds(self):

        # For each of the "10" feature matrices run "5" different data splits:
        for i_mat in range(self.i_mat, self.num_mats_take):
            save_path_mat = self.save_path / Path("Random_mat_{}".format(i_mat + 1))
            
            # Create one split of Train-Test for with one feature matrix for each recording:
            data_class = self.load_data(i_mat=i_mat)

            # Stratified K-folds: Takes group information into account to avoid building folds with imbalanced class distributions
            kf = StratifiedKFold(n_splits=self.k_folds, random_state=self.random_state, shuffle=True)  
            fold = 0
            for train_idx, test_idx in kf.split(data_class.X, data_class.y):  # for each fold:
                X_train, X_test = data_class.X[train_idx], data_class.X[test_idx]
                y_train, y_test = data_class.y[train_idx], data_class.y[test_idx]
                CK_train, CK_test = data_class.recs_ids[train_idx], data_class.recs_ids[test_idx]
                
                # Check if min y of train is not bigger than min y of test, same in max y:
                if min(y_train) > min(y_test):
                    print('ERROR: minimum value of test is smaller than in train')
                if max(y_train) < max(y_test):
                    print('ERROR: Maximum value of test is bigger than in train')

                print('*' * 50)
                print("Train shape: X = {}, y = {}".format(X_train.shape, y_train.shape))
                print("Test shape: X = {}, y = {}".format(X_test.shape, y_test.shape))
                print('*' * 50)
                # Create dictionary with the datasets:
                datas = {
                    'train': {'X': X_train, 'y': y_train, 'recs_ids': CK_train},
                    'test': {'X': X_test, 'y': y_test, 'recs_ids': CK_test}}

                # The path where to save the results of the fold:
                save_path_fold = save_path_mat / r"Trial{}".format(fold + 1)
                print("Save path of the fold: {}".format(save_path_fold))
                # if os.path.isdir(save_path_fold):  # if the folder already exists then continue to next fold
                #     fold += 1
                #     continue

                # Normalize the data:
                norm_datas = self.norm_data(datas)

                # # Hyper tune parameters:
                self.hyper_tune(data_train=norm_datas["train"], save_path=save_path_fold)
                save_path_stat = save_path_fold / Path("Best_{}".format(self.statistic))  
                if not os.path.isdir(save_path_stat):
                    os.makedirs(save_path_stat)
                # Train the model:
                model, history = self.train_mdl(norm_datas["train"])
                # Test data:
                datas_resutls, datas_y = self.test_data(model, norm_datas)
                K.clear_session()
                # Save results into csv file:
                self.save_run_results_to_csv(i_mat, fold, datas_resutls)
                # Save for the splits: 
                self.save_for_split(datas_y, datas_resutls, datas, model, history, save_path_stat)
                del model, history, datas_resutls, datas_y
                plt.close('all')
                print("Done fold {} out of {}".format(fold + 1, self.k_folds))
                fold += 1
                # --------------------------- End of split ------------------------------------ 
            # Close all open plots:
            plt.close('all')
            # --------------------------- End of mat ------------------------------------ 
        # Save the mean results of all mats: 
        print('**Saving summary of all mats...')
        # Save mean results of all datas and all statistics into one excel file:
        self.save_allDatas_allStatistics_excel()
        print('Done')

    # =============================================================================================
    # @profile
    def norm_data(self, datas):
        print('*Normalizing data...')
        norm_datas = {key: {} for key in datas.keys()}

        for key in datas.keys():
            if key == 'train':
                norm_datas[key]["X"], self.transformer = norm_data_by_mat(
                    datas[key]["X"], self.norm_method)
            else:
                norm_datas[key]["X"] = norm_data_by_mat(datas[key]["X"],
                                                        self.norm_method,
                                                        self.transformer)
            norm_datas[key]["y"] = datas[key]["y"].copy()  # copy added on 250523

            print('**Normmed by mat**')
        return norm_datas

    # =============================================================================================
    # @profile
    def train_mdl(self, norm_train):
        print("*Training model...")
        # Train the model with recordings with 1 session:
        input_shape = norm_train["X"][0].shape
        mdl_define = Models(self.best_params["learn_rate"], input_shape,
                            self.model_name, self.params_config.get('n_out'),
                            self.params_config.get('act_out'),
                            [self.params_config.get('loss_func')],
                            [self.params_config.get('metric')])
        mdl_define.define_mdl()
        print("model defined")
        model = mdl_define.model
        # Normalize the target y to be in range [0,1]:
        y_train_norm = np.asarray(
            norm_train["y"]) / self.max_y  # changed to max of y_train.# self.max_score
        ### Shuffle the samples
        X, y = shuffle(norm_train["X"], y_train_norm)
        
        callbacks = []
        if self.params_config.get('early_stopping')['eval']:
            if self.params_config.get('valid_ratio') > 0:  # if there is a validation set
                early_stopping = EarlyStopping(monitor='val_loss',
                                               patience=self.params_config.get('early_stopping')[
                                                   'patience'],
                                               # the number of epochs with no improvement
                                               restore_best_weights=True)  # Whether to restore model weights from the epoch with the best value
            else:
                early_stopping = EarlyStopping(monitor='loss',
                                               patience=self.params_config.get('early_stopping')[
                                                   'patience'],
                                               restore_best_weights=True)
            callbacks = [early_stopping]
            ### The validation data is selected from the last samples in the x and y data provided, BEFORE shuffling
        history = model.fit(X, y,
                            epochs=self.best_params["epochs"],
                            batch_size=self.best_params["batch_size"],
                            verbose=2,  # (0 = silent, 1 = progress bar, 2 = one line per epoch).
                            shuffle=1,  # whether to shuffle the training data before each epoch
                            validation_split=self.params_config.get('valid_ratio'), 
                            callbacks=callbacks)  
        ### "The validation data is selected from the last samples in the x and y data provided, before shuffling. will not train on it."
        print("*Done training")
        return model, history

    # =============================================================================================
    # @profile
    def test_data(self, model, norm_datas):
        datas_results, datas_y = {}, {}
        print("*Testing data...")
        # Predict for each dataset:               
        for key in norm_datas.keys():
            results_train, ys_train = predict_data(model, norm_datas[key]["X"],
                                                   norm_datas[key]["y"],
                                                   self.max_y, self.min_y)
            datas_results[key] = results_train
            datas_y[key] = ys_train

        # Print results:
        for key, res in datas_results.items():
            print('**{}: RMSE = {:.4}, NRMSE = {:.4}, R = {:.4}, R_spear = {:.4}'.format(
                key, res["RMSE"], res["NRMSE"], res["R"], res["R_spear"]))

        return datas_results, datas_y

    # =============================================================================================
    def save_for_split(self, datas_ys, results, datas, model, history, save_path):
        # Save pred-true txt file for each dataset:
        for key, ys in datas_ys.items():
            self.save_pred_true(ys, datas[key]["recs_ids"], save_path, '_' + key)  

        # Save parameters and results:
        self.save_parameters_results(save_path, model, history, results)

        if self.params_config["save_model"] == True:
            # Save the normalization transformer:
            # np.savez(save_path / Path("Transformer.npz"), transformer = self.transformer)
            with open(save_path / Path("Transformer.pkl"), 'wb') as f:
                pickle.dump([self.transformer], f)
            # Save the model:
            self.save_model(save_path, model)

        # Plot the loss function:
        fig_loss = plot_loss(history, title='')
        fig_loss.savefig(save_path / Path("Loss_fig.png"), dpi=self.dpi, bbox_inches='tight')

    # =============================================================================================
    @staticmethod
    def save_pred_true(ys, CKs, save_path, sub_name):
        # # Write predicted and ture scores into a txt file:
        df = pd.DataFrame({'rec_id': np.squeeze(CKs),
                           'y_pred': np.squeeze(ys['y_pred']),
                           'y_true': np.squeeze(ys['y_true'])})
        df.to_csv(save_path / Path("Pred_true" + sub_name + ".txt"),
                  sep='\t', na_rep="none", index=False)

    # =============================================================================================
    def save_parameters_results(self, save_path, model, history, results):
        f = open(save_path / Path("Parameters&Results.txt"), "w")
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        f.writelines(short_model_summary)
        f.writelines('\nBatch size = {0}'.format(self.best_params["batch_size"]))
        f.writelines('\nlearn rate = ' + str(self.best_params["learn_rate"]))
        f.writelines('\nNum epochs = ' + str(len(history.history[
                                                     'loss'])))  # Because of the early stopping.
        for key, res in results.items():
            f.writelines('\n\n{}:\n'.format(key))
            f.writelines('{}:{}\n'.format(k, v) for k, v in res.items())
        f.close()

    # =============================================================================================
    def plot_pred_true_per_data(self, datas):
        fig = plt.figure(figsize=(15, 15))
        i = 1
        for key, ys in datas.items():
            max_score = np.max((ys["y_pred"], ys["y_true"]))
            min_score = np.min((ys["y_pred"], ys["y_true"]))
            plt.subplot(2, 3, i)
            plot_pred_true(
                ys["y_pred"], ys["y_true"],
                max_score, min_score,
                xlabel='Predicted {}'.format(self.target_score),
                ylabel='Observed {}'.format(self.target_score),
                title=key, fig=fig)
            i += 1
        return fig

    # =============================================================================================   
    def save_run_results_to_csv(self, i_mat, i_split, datas):
        '''Save the mean results of different mats and splits'''

        for data_name, results in datas.items():
            ## Concatenate the parameters and their results to one dict:
            csv_row = {'Feat_mat': i_mat, 'Split': i_split}
            csv_row.update(results)  # add to the same row
            df = pd.DataFrame([csv_row])
            file_name_path = self.save_path_statistic / Path("Run_results_{}.csv".format(data_name))

            if os.path.isfile(file_name_path):  # if the files already exists then:
                all_data = pd.read_csv(file_name_path)
                row_id = all_data[
                    (all_data['Feat_mat'] == i_mat) & (all_data['Split'] == i_split)].index.tolist()
                if row_id:  # if the i_mat and i_split exist then replace the results:
                    all_data.loc[row_id[0]] = csv_row
                    all_data.to_csv(file_name_path, mode='wb', index=False, header=True)
                    # header=True because it writes all results from the beginning. and not just adding a row
                    print('Replaced row in the results file')
                else:  # if this run is new then append at the end
                    df.to_csv(file_name_path, mode='a', index=False, header=False)
            else:  # if the files doesn't exist then create the file and write into it:
                df.to_csv(file_name_path, mode='w', index=False, header=True)
    
    # =============================================================================================
    def read_results_dataset_n_mats(self, df):  

        analyzed_mats = np.unique(df['Feat_mat'])
        results_per_mat = [dict() for _ in range(max(analyzed_mats) + 1)]  # list of dictionaries

        for i_mat in analyzed_mats:  # for each feature matrix
            results_per_mat[i_mat] = self.read_results_dataset_i_mat(df, i_mat)
        return results_per_mat

    # =============================================================================================   
    def read_results_dataset_i_mat(self, df, i_mat):  
        '''
        df: DataFrame with results for each i_mat, i_split, and statistic
        i_mat: the feature matrix number to extract the results
        
        output:
            i_mat: a dictionary, where each key is a statistic,
                   and the the values is a list of the results
        '''
        results = dict()
        results_i_mat = df.loc[df['Feat_mat'] == i_mat].drop(columns=['Feat_mat', 'Split'])
        for statistic_name, values in results_i_mat.items():  # for each statistic
            results[statistic_name] = values.tolist()

        return results

    # =============================================================================================
    def read_results_all_dataset(self):
        results_datas = dict()
        for dataset_name in self.dataset_names:  # for each dataset
            df = self.load_all_results(dataset_name)
            results_datas[dataset_name] = self.read_results_dataset_n_mats(df)
        return results_datas

    # =============================================================================================   
    def load_all_results(self, dataset_name):  
        '''
        dataset_name: string. 
                      the name of the dataset (train, test,...)
        output:
            results_per_mat: a list of dictionaries, where each dict is for different feature martix,
            and each key in the dict is a statistic, and the the values is a list of the results
        '''
        file_name_path = self.save_path_statistic / Path("Run_results_{}.csv".format(dataset_name))
        df = pd.read_csv(file_name_path)
        df = df.sort_values(by=['Feat_mat', 'Split'])
        return df

    # =============================================================================================   
    def save_allDatas_allStatistics_excel(self):  
        results_datas = self.read_results_all_dataset()  # dict of lists of dicts

        my_dict = {}
        for data_name in self.dataset_names:
            results_data = results_datas[data_name]
            mean_data = {}
            for statistic in results_data[0].keys():
                values = functools.reduce(operator.iconcat, [d[statistic] for d in results_data if
                                                             statistic in d.keys()], [])
                mean_data[statistic] = "{:.3f} \u00B1 {:.3f} ({:.3f})".format(
                    np.nanmean(values), np.nanstd(values, ddof=1),
                    np.nanmedian(values))  # mu +- sd (median)
                my_dict[data_name] = mean_data

        df = pd.DataFrame(my_dict)
        df = df.transpose()
        filename = self.save_path_statistic / 'Mean_allDatas_allStatistics.xlsx'
        df.to_excel(filename, index=True)

    # =============================================================================================           
    def save_model(self, save_path, model):  
        model_json = model.to_json()  
        with open(save_path / Path('Model.json'), "w") as json_file:
            json_file.write(model_json)
        model.save_weights(save_path / Path('Model_weights.h5'))