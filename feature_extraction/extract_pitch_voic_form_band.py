# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:24:12 2022

@author: marinamu
"""

# ######################################################################
# python extract_voicing_from_wav.py in.wav out.txt
# ######################################################################

import math
from utils import my_writer, write_line, to_string
import parselmouth
from optparse import OptionParser

# PITCH_FLOOR = 60
# PITCH_CEILING = 1600


# set_parser
# =================================================================================================
def get_parser():
    parser = OptionParser()
    parser.add_option("-r", "--rec_name", dest="rec_name",
                      help="The name of the recording", action="store")
    parser.add_option("-o", "--orig_path", dest="orig_path",
                      help="The name of the recording", action="store")
    parser.add_option("-s", "--save_path", dest="save_path",
                      help="The name of the pitch file name", action="store")
    parser.add_option("-pf", "--pitch_floor", dest="pitch_floor",
                      help="The lowest pitch value", action="store")
    parser.add_option("-pc", "--pitch_ceiling", dest="pitch_ceiling",
                      help="The highest pitch value", action="store")
    
    return parser


# =================================================================================================

def main(param_dict):
    orig_path = param_dict["orig_path"]
    save_path = param_dict["save_path"]
    rec_name = param_dict["rec_name"]
    pitch_floor = param_dict.get("pitch_floor", 60)
    pitch_ceiling = param_dict.get("pitch_ceiling", 1600)
    
    out_txt1 = "{}\\pitch_{}".format(save_path, rec_name.replace("wav", "txt"))
    out_txt2 = out_txt1.replace("pitch", "voicing")
    out_txt3 = out_txt1.replace("pitch", "formants")

    snd = parselmouth.Sound("{}\\{}".format(orig_path, rec_name))
    w1 = my_writer(out_txt1)  # f0
    w2 = my_writer(out_txt2)  # voicing
    w3 = my_writer(out_txt3)  # formants and bandwidths

    print("Running pitch and voicing extraction for {}".format(rec_name))

    my_pitch = snd.to_pitch(time_step=0.01, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    for i in range(1, my_pitch.n_frames + 1):
        frame = my_pitch.get_frame(i)
        f0 = frame.selected.frequency
        strength = frame.selected.strength

        if math.isnan(strength):
            strength = 0
        if math.isnan(f0):
            f0 = 0

        write_line(w1, str(f0))
        write_line(w2, str(strength))
    w1.close()
    w2.close()

    print("Running formants and bandwidths extraction for {}".format(rec_name))
    my_formants = snd.to_formant_burg(time_step=0.01, window_length=0.04)
    for t in my_pitch.xs():
        f1 = my_formants.get_value_at_time(1, t)
        f2 = my_formants.get_value_at_time(2, t)
        bw1 = my_formants.get_bandwidth_at_time(1, t)
        bw2 = my_formants.get_bandwidth_at_time(2, t)
        write_line(
            w3, to_string(f1, 2) + "\t" + to_string(f2, 2) + "\t" + to_string(bw1, 2) + "\t" + to_string(bw2, 2))
    w3.close()


# =================================================================================================  
if __name__ == "__main__":
    ''' To run the script for a recording from CMD:
    python extract_pitch_voic_form_band.py -o <path of the rec> -r <rec_name> -s <save_path>  -pf 60 -pc 1600
    '''
    parser = get_parser()
    (options, args) = parser.parse_args()  # when running through CMD
    param_dict = dict()
    param_dict["rec_name"] = options.rec_name
    param_dict["orig_path"] = options.orig_path
    param_dict["save_path"] = options.save_path
    param_dict["pitch_floor"] = options.pitch_floor
    param_dict["pitch_ceiling"] = options.pitch_ceiling

    main(param_dict)
