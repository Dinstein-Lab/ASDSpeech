# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:24:12 2022

@author: marinamu
"""

import os
from pathlib import Path
import pandas as pd

os.chdir(Path(__file__).parent.absolute())

from extract_pitch_voic_form_band import main

path_recs = r'D:\Autism\Database\recs_no_pitch'
save_path = r'D:\Autism\Database\Pitch_formants_voicing'  # path_recs
pitch_floor = 60 # Hz
pitch_ceiling = 1600 # Hz
time_step = 0.01 # sec
window_length = 0.04 # sec

list_recs_file = []
# %% 
if not list_recs_file:
    # Get the names of the recordings (including.wav):
    recs = [f for f in os.listdir(path_recs) if f.endswith('.wav')]
else:
    recs_list = pd.read_csv(list_recs_file, sep="\t", header=None)
    recs = [f"{rec[0]}.wav" for _,rec in recs_list.iterrows()]
    
# Create the folder where to save the txt files:
if not os.path.isdir(save_path):
    os.mkdir(save_path)
# For each recording extract 3 txt files:
## pitch_{rec name}.txt, formants_{rec name}.txt, voicing_{rec name}.txt:
for rec in recs:
    args = {'orig_path': path_recs,
            'rec_name': rec,
            'save_path': save_path,
            'pitch_floor': pitch_floor,
            'pitch_ceiling': pitch_ceiling,
            'time_step': time_step,
            'window_length': window_length}
    

    main(args)
# %%
''' To run the extract_pitch_voic_form_band.py for a recording from CMD:
python extract_pitch_voic_form_band.py -o <path of the rec> -r <rec_name> -s <save_path> -pf 60 -pc 1600
'''
# %% Organize:
''' 
put all pitch .txt files into Pitch folder,
all foramtns .txt files into Formants folder, and
all voicing .txt files into Voicing folder.    
'''
# Create the three folders:
pitch_folder = "{}\\Pitch".format(save_path)
formants_folder = "{}\\Formants".format(save_path)
voicing_folder = "{}\\Voicing".format(save_path)

if os.path.isdir(pitch_folder) == False:
    os.mkdir(pitch_folder)
if os.path.isdir(formants_folder) == False:
    os.mkdir(formants_folder)
if os.path.isdir(voicing_folder) == False:
    os.mkdir(voicing_folder)

for rec in recs:
    os.replace("{}\\pitch_{}".format(save_path, rec.replace("wav", "txt")),
               "{}\\pitch_{}".format(pitch_folder, rec.replace("wav", "txt")))
    os.replace("{}\\formants_{}".format(save_path, rec.replace("wav", "txt")),
               "{}\\formants_{}".format(formants_folder, rec.replace("wav", "txt")))
    os.replace("{}\\voicing_{}".format(save_path, rec.replace("wav", "txt")),
               "{}\\voicing_{}".format(voicing_folder, rec.replace("wav", "txt")))
