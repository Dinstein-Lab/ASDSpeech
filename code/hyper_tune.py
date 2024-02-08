# -*- coding: utf-8 -*-
"""
Created on 05.01.2022.

@author: marinamu
"""
import numpy as np
import os
import random
import itertools
import pandas as pd

random.seed(1337)
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import backend as K

from train_test_1split import TrainTest1Split
from tic_toc_class import tic_toc
from commons_functions import norm_data_by_mat

class HyperTune:
    
    def __init__(self, tuning_params, params_config, hyper_tune_config):
        self.X = tuning_params["X"]  # the whole dataset
        self.y = tuning_params["y"]  # target vector
        self.param_grid = tuning_params["param_grid"]  # dict of the parameteres to tune
        self.save_path = tuning_params["save_path"]
        self.data_filename = tuning_params["data_filename"]
        self.cv = hyper_tune_config["cv_k"]  # number of kfolds in cross validation
        self.n_iters = hyper_tune_config["n_iters"]  # in RandomizedSearchCV: number of combinations to run
        self.params_idx = hyper_tune_config["params_idx"]
        self.statistic = hyper_tune_config["statistic"]  # the statistic to sort by/ choose the best CV: "RMSE"
        self.model_name = params_config["model_name"]
        self.norm_method = params_config["norm_method"]
        self.score_type = params_config["score_name"]  # ADOS

        self.params_config = params_config

        self.rec_idx_X = np.arange(0, len(self.y), step=1)  # default.
        self.prepare_data_TF = True
        self.search = None
        self.search_results = None
        self.random_state = 0  # random state of the data split

    # --------------------------------------------------------------------------------------------#
    def run_all(self):
        if self.prepare_data_TF:
            print('\nPrepare data.')
            self.prepare_data()
        else:  # Do not normalize:
            self.X_norm = self.X.copy()
        print('\nRun the hyperparameter tunning process:')

        print('*' * 80)
        self.run_random_search_manual()
        print('*' * 80)
        print('\nGet best results.')
        self.load_best_params()
        print('*' * 80)
        self.summarize_results()

    # --------------------------------------------------------------------------------------------#
    def prepare_data(self):
        # Normalize each matrix of a recording by its max value (per feature)
        self.X_norm, self.transformer = norm_data_by_mat(self.X, self.norm_method)
        print('*' * 50)
        print("Train shape: X = {}, y = {}".format(self.X_norm.shape, self.y.shape))
        print('*' * 50)

    # --------------------------------------------------------------------------------------------#
    def run_random_search_manual(self):
        """
        1. Create different combinations of the hyper-parameters.
        2. Choose random "n_iters" combinations: shuffle the combinations and choose the first "n_iters" combinations.
        3. For each set of parameters / combination:
            a. Divide Train to Train-Validation "cv" times, where all the matrices of each recording are in the same group.For each CV:
                i. Train the model using the new Train set.
                ii. Validate the model using the Validation set -> Calculate the RMSE, R, p
                iii. Save the results
            b. Calculate mean RMSE for all the cross validations. Save the results.
        4. Choose the set of parameters with the lowest mean RMSE value.
        """
        flag = 0
        # 1:
        keys = self.param_grid.keys()
        values = (self.param_grid[key] for key in keys)
        params_combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        # 2.
        random.shuffle(params_combinations)  # shuffles the params_combinations itrself
        self.params_sets_run = params_combinations[
                               0:min([self.n_iters, len(params_combinations)])]  # choose first "n_iters"s to run on
        # 3. 
        self.mean_cv_results = []
        self.sd_cv_results = []
        self.mean_cv_all_results = []
        for params_idx in range(self.params_idx - 1, self.n_iters):  # 12.03.21.
            tune_params = self.params_sets_run[params_idx]
            print('\nNow Evaluating {}/{}: {}'.format(params_idx + 1, self.n_iters, tune_params))
            cv_results = []
            cv_all_results = []
            # Create a folder for the results of this set of parameters:
            kf = StratifiedKFold(n_splits=self.cv, random_state=1337, shuffle=True)
            for train_idx, valid_idx in kf.split(self.X_norm, self.y):  # for each iteration of cross validation:
                # 3.a. Split to train-validation:
                X_train_norm, X_valid_norm = self.X_norm[train_idx], self.X_norm[valid_idx]
                y_train, y_valid = self.y[train_idx], self.y[valid_idx]
                print('*' * 50)
                print("Train shape: X = {}, y = {}".format(X_train_norm.shape, y_train.shape))
                print("Validation shape: X = {}, y = {}".format(X_valid_norm.shape, y_valid.shape))
                print('*' * 50)

                data = {"X_train": X_train_norm, "y_train": y_train,
                        "X_test": X_valid_norm, "y_test": y_valid,
                        "data_file_name": self.data_filename, "save_path": self.save_path}
                run_mdl_params = {**self.params_config, **tune_params}  # merge 2 dicts. if overlapping keys -> the value from tune_params is taken

                # 3.a.i. Create the model class:
                mdl_class = TrainTest1Split(data, run_mdl_params)

                # 3.a.i. Run the training process
                print('\n-Training model...')
                tic_toc.tic()
                mdl_class.create_mdl()
                mdl_class.train_mdl()
                # 3.a.ii. Evaluate the valid and train datasets:
                print('-Done training. Evaluating data...')
                mdl_class.evaluate_mdl()
                print(f"Time took: {tic_toc.toc()}")
                # 3.a.iii. Save the perforamnce value (RMSE or Balanced Accuracy):
                cv_results.append(mdl_class.results_test[self.statistic])
                cv_all_results.append(mdl_class.results_test)
                # Save models layers summary to txt file only once:
                if flag == 0:
                    self.save_model_layers(model=mdl_class.model)
                    flag += 1
                K.clear_session()
            print("cv_results: {}".format(cv_results))
            # 3.b. Save mean perforamnces of the cross validations for this set of parameters:
            self.mean_cv_results.append(round(np.mean(cv_results), 4))
            self.sd_cv_results.append(round(np.std(cv_results, ddof=1), 4))
            mean_cv_all_results_set = dict()
            for key_name in cv_all_results[0].keys():
                mean_cv_all_results_set[key_name] = round(np.nanmean([dic[key_name] for dic in cv_all_results]), 4)

            self.mean_cv_all_results.append(mean_cv_all_results_set)
            print("Mean results:\n{}: \nMean = {}".format(tune_params, mean_cv_all_results_set))
            self.save_meanCV_to_csv(tune_params, mean_cv_all_results_set)  # save to a file the parameters and their results
            print('Done Evaluating {}/{}.'.format(params_idx + 1, self.n_iters))

    # --------------------------------------------------------------------------------------------#
    def load_best_params(self):
        if self.statistic in ['RMSE', 'NRMSE']:
            ascending = True  # the lower = the best
        elif self.statistic in ['R', 'CCC']:
            ascending = False  # the higher = the best

        file_name_path = self.save_path / Path("Mean_CV_results.csv")
        df = pd.read_csv(file_name_path)
        # Sort the table:
        ## save the sorted into the same df var. if inplace=False then you need to save to a new var
        df.sort_values([self.statistic], axis=0, ascending=[ascending], inplace=True)
        self.best_params = df.iloc[0].to_dict()  # {'row1': {'col1': 1, 'col2': 0.5}, 'row2': {'col1': 2, 'col2': 0.75}}
        print("\nBest parameters: {}".format(self.best_params))

    # --------------------------------------------------------------------------------------------#
    def save_meanCV_to_csv(self, params, mean_cv_all_results):
        '''Save the mean results of the cross validation of one set of parameters'''
        # Save the mean results to CSV file:
        ## Concatenate the parameters and their results to one dict:
        csv_row = {}
        csv_row.update(params)
        csv_row.update(mean_cv_all_results)
        df = pd.DataFrame.from_dict([csv_row], orient='columns')
        file_name_path = self.save_path / Path("Mean_CV_results.csv")
        if os.path.isfile(file_name_path):  # if the files already exists then append at the end:
            df.to_csv(file_name_path, mode='a', index=False, header=False)
        else:  # if the files doesn't exist then create the file and write into it:
            df.to_csv(file_name_path, mode='w', index=False)

    # --------------------------------------------------------------------------------------------#
    def summarize_results(self):
        print("\nBest parameters: {}".format(self.best_params))

    # --------------------------------------------------------------------------------------------#  
    def save_model_layers(self, model):
        file_name_path = self.save_path / Path("Model_summary.txt")
        if not os.path.isfile(file_name_path):  # if the file doesnt exists then create:
            f = open(file_name_path, "w")
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            short_model_summary = "\n".join(stringlist)
            f.writelines(short_model_summary)
            f.close()