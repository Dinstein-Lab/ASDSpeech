# Quantifying longitudinal changes in core autism symptoms using automated speech analysis

The aim is to estimate autism severity from audio signals. This is done by training a convolutional neural network on Autism Diagnostic Observation Schedule 2nd edition (ADOS-2) recordings of 136 children using acoustic and conversational hand-crafted features. The algorithm is tested on independent recordings from 61 additional children who completed two ADOS-2 assessments, separated by 1-2 years.

## Folders organization
•	`./code`: python files that are used to train the model (`.code/main_script.py`) and test the two time-points datasets (`./code/estimate_recs_trained_mdl.py`).

•	`./config`: the configuration files: `./config/config_file.yaml `used for training and `./config/config/config_file_trained_mdl.yaml` used for testing on the 61 children.

•	`./data`: includes .mat (Matlab) file of the extracted features of the training datasets (`./data/train_data.mat`) and separate .mat files for all 122 recordings in the test datasets in the format <rec_name>.mat, the lists of recordings names of the 61 children in the test datasets (`./data/data_T1.yaml` and `./data/data_T2.yaml`), and the Excel files of the test datasets that include children characteristics (age, severity scores, gender. `./data/data_T1_info.xlsx` and `./data/data_T2_info.xlsx`).

•	`./results`: includes all the trained models for each target score (SA, RRB, total ADOS) for each iteration (model and weights) and the estimated scores (.txt files).

## Data
The database included in this work consists of 136 children who participated in a single ADOS-2 assessment and 61 children who participated in two ADOS-2 assessments separated by 1–2 years (two recordings each), yielding 258 ADOS-2 assessments in total. We extracted 49 features of speech from each ADOS-2 recording. These included acoustic features (e.g., pitch, jitter, formants, bandwidth, energy, voicing, and spectral slope) and conversational features (e.g., mean vocalization duration and total number of vocalizations). All ADOS-2 assessments were performed by a clinician with research reliability. In addition, all participating children had ASD diagnoses that were confirmed by both a developmental psychologist and either a child psychiatrist or a pediatric neurologist, according to the Diagnostic and Statistical Manual of Mental Disorders, Fifth Edition (DSM-5) criteria. Informed consent was obtained from all parents, and the SUMC Helsinki committee approved the study.

## Recording setup

All recordings were performed during ADOS-2 assessments using a single microphone (CHM99, AKG, Vienna) located on a wall, ~1–2m from the child, and connected to a sound card (US-16x08, TASCAM, California). Each ADOS-2 session lasted ~40 minutes (40.75 ± 11.95 min) and was recorded at a sampling rate of 44.1 kHz, 16 bits/sample (down-sampled to 16 kHz). The audio recordings were manually divided and labeled as child, therapist, parent, simultaneous speech (i.e., speech of more than one speaker), or noise (e.g., movements in the room) segments. All remaining segments were automatically labeled as silent. Only child-labeled segments were used in further analysis. These segments included speech, laughing, moaning, crying, and screaming.

## Feature extraction

We extracted 49 features that were categorized into nine groups: pitch, formants, jitter, voicing, energy, Zero-Crossing Rate (ZCR), spectral slope, duration, and quantity/number of vocalizations. All features, except duration and quantity, were first extracted in 40ms windows (window overlap of 75%), resulting in a vector of feature values per vocalization. The minimum, the maximum, and the mean pitch of the voiced vocalizations (across windows) were computed, deriving one value for each vocalization. We then selected a group of 10 consecutive vocalizations and computed the mean and variance across vocalizations for relevant features. We also computed the mean duration of vocalizations and the overall number of vocalizations in the recording. These steps yielded a vector with 49 values corresponding to the 49 features per 10 vocalizations. We performed this procedure 100 times, selecting random groups of ten consecutive vocalizations from the recording. Combining these 100 samples yielded a features matrix of 100×49 per child, with the last column (quantity of vocalizations) containing the same value across all rows. 

## Data pre-processing

The features (feature matrix of size 100x49 per recording) are normalized using the z-norm method (zero mean and unit variance), where the normalization is applied per feature on the whole train dataset, and the same mean and standard deviation are used to normalize the test datasets. In this paper, we applied five feature matrices for each child.

## Training

This process includes 5-fold cross-validation for each of the feature matrices, where each fold includes hyper-parameters tuning (learning rate, batch size, and number of epochs), testing the best parameters on the fifth fold, and performance evaluation using Pearson correlation, Root Mean Squared Error, Normalized RMSE, and Concordance Correlation Coefficient.

Code to run this part: `main_script.py`.

## Estimation/Prediction

This process includes the application of each trained model of each fold (5x5=25 models in total) on the two test datasets: Tim-point 1 and time-point 2. This step includes the estimation of the target score of each recording in time-point and performance evaluation.

The code to run this part is `estimate_recs_trained_mdl.py`.

# Run
To make it run properly, clone this repository in a folder.
From your command line, go to ASE_audio/code folder and run the following python scripts:

**Training**
``` python
# Run training using the configuration file
python main_script.py -c ../config/config_file.yaml
```
**Estimation/Prediction**
``` python
# Run testing using the configuration file
python code/estimate_recs_trained_mdl.py -c config/config_file_trained_mdl.yaml
```
