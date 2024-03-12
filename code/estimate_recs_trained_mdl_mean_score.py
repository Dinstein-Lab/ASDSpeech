# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:50:13 2023

@author: MARINAMU

ASE on different recordings based on trained model
"""

from tensorflow.keras.models import model_from_json
import pickle
from scipy.io import loadmat
import numpy as np
from tensorflow.keras import backend as K
from pathlib import Path
import pandas as pd
from scipy import stats
import os
import yaml
from optparse import OptionParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # dont print messages

mainpath = Path(__file__).parent.absolute()
os.chdir(mainpath)

from find_min_max_score import FindMinMaxScore
from sort_human import natural_keys
from calc_ccc_metric import CalcCCC


# set_parser
# =================================================================================================
def set_parser():
    parser = OptionParser()
    parser.add_option("-c", "--config", dest="config",
                      help="The configuration file that will be executed")
    parser.add_option("-d", "--debug", dest="debug", action='store_true',
                      help="Run in debug mode")

    (options, args) = parser.parse_args()
    return options


# Load the model and its weights:
# =================================================================================================
def load_model_weights(path):
    # Load the trained model and its weights:
    json_file = open(path / "Model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(path / "Model_weights.h5")
    return model


# Get the current script's directory
SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
# Get the project root directory
PROJECTS_DIR = Path(os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir)))


# load_yaml
# =============================================================================
def load_yaml(file_pointer):
    with open(file_pointer) as file:
        return yaml.full_load(file)


# Main script:
# =============================================================================
if __name__ == "__main__":
    # Load the yaml file:
    options = set_parser()
    config_dict = load_yaml(file_pointer=options.config)

    # Estimate recordings using a trained model:
    #  load model and the weights:
    data_files_path = PROJECTS_DIR / r'data'
    results_path = PROJECTS_DIR / r"results" / config_dict["trained_mdl_path"]
    target_score = os.path.basename(os.path.join(str(results_path).split("_prediction")[0]))
    statistic = f"Best_{config_dict['statistic']}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config_dict["GPU_id"])
    save_results_pred_true = config_dict["save_results_pred_true"]

    evals_df = [pd.read_excel(data_files_path / Path(rf"data_T{i + 1}_info.xlsx"))
                for i in range(2)]
    # Concatenate two tables one below other:
    df_all = pd.concat(evals_df, ignore_index=True)

    # Get the min and max values of the target score:
    min_score, max_score = FindMinMaxScore(target_score_name=target_score).calculate_min_max()
    # Get a list of folder names in the specified directory
    folder_names = [folder for folder in os.listdir(results_path) if
                    os.path.isdir(os.path.join(results_path, folder))]
    # Filter folders that start with "Random_mat
    mat_folders = [folder for folder in folder_names if folder.startswith("Random_mat")]
    mat_folders.sort(key=natural_keys)

    test_sets = ["data_T1", "data_T2"]
    results_test = {test_set: [] for test_set in test_sets}

    test_sets_df = {}
    # Load the test recs names:
    for test_set in test_sets:
        # Load the list of recordings of a time-point:
        test_set_recs = load_yaml(file_pointer=data_files_path / rf'{test_set}.yaml')
        # Create a dataframe with rec_id:
        test_sets_df[test_set] = pd.DataFrame({"rec_id": test_set_recs})
    i_iter = 1  # idx of the running iteration (model)

    dfs_true_pred = {time_point: [] for _, time_point in test_sets}
    # Run for each Random_mat (feature mat):
    for mat_folder in mat_folders:
        # Get the number of the feature mat:
        i_mat = int(''.join(filter(str.isdigit, mat_folder)))  # 0,1,2...
        # The full path:
        full_mat_path = results_path / mat_folder
        # Get a list of folder names in the specified directory
        folder_names = [folder for folder in os.listdir(full_mat_path) if
                        os.path.isdir(os.path.join(full_mat_path, folder))]
        # Filter folders that start with Trial/Fold:
        folds_folders = [folder for folder in folder_names if
                         folder.startswith("Trial") or folder.startswith("Fold")]
        folds_folders.sort(key=natural_keys)
        # Run for each Fold:
        for fold_folder in folds_folders:
            # Get the number of the fold:
            i_fold = int(''.join(filter(str.isdigit, fold_folder)))  # 0,1,2...
            # The full path:
            dest_path = full_mat_path / fold_folder / statistic
            # Load the trained model:
            model = load_model_weights(path=dest_path)
            print("Trained model loaded.")
            # Load normalization transformer:
            with open(dest_path / "Transformer.pkl", 'rb') as f:
                transformer = pickle.load(f)[0]

            true_pred_score_df = {test_set: pd.DataFrame() for test_set in test_sets}
            for test_set in test_sets:

                test_df = test_sets_df[test_set]
                test_info_df = pd.merge(test_df, df_all,
                                        left_on=['rec_id'], right_on=['rec_id'], how='left')
                true_pred_score = []
                true_pred_score_loaded = []
                for i_rec, row in test_info_df.iterrows():
                    # load features of a recording from the test dataset:
                    feat_mat_rec = loadmat(data_files_path / f"{row['rec_id']}.mat",
                                           variable_names=["features"])
                    print("features file of rec {i_rec}/{len(test_info_df)} was loaded")
                    # Take the i_mat feature matrix:
                    features = np.asarray(feat_mat_rec["features"][i_mat - 1])[0]
                    if features.any():
                        # Target score:
                        score = int(row[target_score])
                        # Normalize the feature matrix using loaded normalization transformer:
                        X_no_nan = np.nan_to_num(features)
                        X_norm = transformer.transform(X_no_nan)
                        # Convert to 3d matrix: 1x49x100
                        X_norm_3d = np.expand_dims(X_norm, axis=0)
                        # Predict score using trained model:
                        score_pred = np.squeeze(np.clip(
                            np.round(model.predict(X_norm_3d, verbose=0) * max_score).astype('int'),
                            min_score, max_score))
                        # Append the results as dict to a list:
                        true_pred_score.append({"rec_id": row["rec_id"],
                                                "y_true": score,
                                                "f'y_pred_{i_iter + 1}'}": score_pred})
                print(f"Done {i_iter} time-point {test_set}")
                # Convert to a long dataframe with all the recordings' predicted and actual values:
                dfs_true_pred[test_set].append(pd.DataFrame.from_dict(true_pred_score))
                # End of a test_set.
            i_iter += 1
            # End of a Fold
            K.clear_session()
        # End of a feature mat.

    # Calculate the mean estimated score for each child and calculate performance:
    merged_df = {time_point: [] for _, time_point in test_sets}
    for time_point in test_sets:
        # Start with the first dataframe
        merged_df[time_point] = dfs_true_pred[time_point][0][['CK', 'Date', 'y_true'] +
                                                             dfs_true_pred[0].filter(like='y_pred_').columns.tolist()]

        # Loop through the remaining dataframes and merge them one by one
        for df in dfs_true_pred[time_point][1:]:
            merged_df[time_point] = pd.merge(merged_df[time_point], df,
                                             on=['CK', 'Date', 'y_true'])
            # Extract columns starting with 'y_pred_'
            y_pred_columns = df.filter(like='y_pred_').columns.tolist()
            
            # Extract columns 'CK', 'Date', 'y_true' and those starting with 'y_pred_'
            selected_columns = ['CK', 'Date', 'y_true'] + y_pred_columns
            
            # Merge DataFrames based on selected columns
            merged_df = pd.merge(merged_df, df[selected_columns], on=['CK', 'Date', 'y_true'])
            
        # Select only the columns with 'y_pred_' prefix for mean calculation
        y_pred_columns = merged_df.filter(like='y_pred_')
        
        # Add a new column 'mean_y_pred' with the mean value across all 'y_pred_' columns
        merged_df['mean_y_pred'] = np.round(y_pred_columns.mean(axis=1))

        # Calculate performance:
        R_pear, p_pear = stats.pearsonr(merged_df[test_set].y_true,
                                        merged_df[test_set].mean_y_pred)
        R_spear, p_spear = stats.spearmanr(merged_df[test_set].y_true,
                                           merged_df[test_set].mean_y_pred)
        rmse = np.sqrt(np.mean((merged_df[test_set].y_true - merged_df[test_set].mean_y_pred) ** 2))
        nrmse = rmse / (max_score - min_score)
        CCC = CalcCCC(merged_df[test_set].y_true,
                      merged_df[test_set].mean_y_pred).calc_metric()
        print(f"r({len(merged_df[test_set])-2}) = {R_pear:.3f}, p-value={p_pear:.4f}")
        print(f"NRMSE = {nrmse:.3f}\nCCC={CCC:.4f}")
