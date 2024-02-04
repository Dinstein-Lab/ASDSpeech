# Quantifying longitudinal changes in core autism symptoms using automated speech analysis

The aim is to estimate autism severity from audio signals. This is done by training a convolutional neural network on Autism Diagnostic Observation Schedule 2nd edition (ADOS-2) recordings of 136 children using acoustic and conversational hand-crafted features. The algorithm is tested on independent recordings from 61 additional children who completed two ADOS-2 assessments, separated by 1-2 years.

## Folders organization
•	`./code`: python files that used to train (`.code/main_script.py`) and test (`./code/estimate_recs_trained_mdl.py`) the algorithm.

•	`./config`: the configuration files: `./config/config_file.yaml `used for training and `./config/config/config_file_trained_mdl.yaml` used for testing on the 61 children.

•	`./data`: includes .mat (Matlab) file of the extracted features of the training datasets (`./data/train_data.mat`) and separate .mat files for all 122 recordings in the test datasets in the format <rec_name>.mat, the lists of recordings names of the 61 children in the test datasets (`./data/data_T1.yaml` and `./data/data_T2.yaml`), and the Excel files of the test datasets that include children characteristics (age, severity scores, gender. `./data/data_T1_info.xlsx` and `./data/data_T2_info.xlsx`).

•	`./results`: includes all the trained models for each target score (SA, RRB, total ADOS) for each iteration (model and weights) and the estimated scores (.txt files).

## Data pre-processing

The features (feature matrix of size 100x49 per recording) are normalized using the z-norm method (zero mean and unit variance), where the normalization applied per feature on the whole train dataset, and the same mean and standard deviation are used to normalize the test datasets. In this paper, we applied five feature matrices for each child.

## Training

This process includes 5-folds cross validation for each of the feature matrix, where each fold includes hyper-parameters tuning (learning rate, batch size, and number of epochs), testing the best parameters on the fifth fold, and performance evaluation using Pearson correlation, Root Mean Squared Error, Normalized RMSE, and Concordance Correlation Coefficient.

Code to run this part: `main_script.py`.

## Estimation/Prediction

This process includes the application of each trained model of each fold (5x5=25 models in total) on the two test datasets: Tim-point 1 and time-point 2. This step includes the estimation of the target score of each recording in in time-point and performance evaluation.

Code to run this part: `estimate_recs_trained_mdl.py`.

# Run
To make it run properly, clone this repository in a folder.
From your command line go to ASE_audio folder and run the following python scripts:

**Training**
``` python
# Run training using the configuration file
python code/main_script.py -c config/config_file.yaml
```
**Estimation/Prediction**
``` python
# Run testing using the configuration file
python code/estimate_recs_trained_mdl.py -c config/config_file_trained_mdl.yaml
```
