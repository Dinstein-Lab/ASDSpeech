**Quantifying longitudinal changes in core autism symptoms using automated speech analysis**

The aim is to estimate autism severity from audio signals. This is done by training a convolutional neural network on Autism Diagnostic Observation Schedule 2nd edition (ADOS-2) recordings of 136 children using acoustic and conversational hand-crafted features. the algorithm is tested on independent recordings from 61 additional children who completed two ADOS-2 assessments, separated by 1-2 years.

Folders organization
•	`./code`: python files that used to train (`.code/main_script.py`) and test (`./code/estimate_recs_trained_mdl.py`) the algorithm.

•	`./config`: the configuration files: `./config/config_file.yaml `used for training and `./config/config/config_file_trained_mdl.yaml` used for testing on the 61 children.

•	`./data`: includes .mat (Matlab) file of the extracted features of the training datasets (`./data/train_data.mat`) and separate .mat files for all 122 recordings in the test datasets in the format <rec_name>.mat, the lists of recordings names of the 61 children in the test datasets (`./data/data_T1.yaml` and `./data/data_T2.yaml`), and the Excel files of the test datasets that include children characteristics (age, severity scores, gender. `./data/data_T1_info.xlsx` and `./data/data_T2_info.xlsx`).

•	`./results`: includes all the trained models for each target score (SA, RRB, total ADOS) for each iteration (model and weights) and the estimated scores (.txt files).


