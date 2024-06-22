import os
import re
import codecs
import numpy as np
from random import randint
#from itertools import imap

def normalize_matrix_by_sum_column(mat):
    for i in range(len(mat[:,0])):
        col_sum = mat[:, i].sum()
        if col_sum != 0:
            mat[:, i] = mat[:, i] / col_sum
        else:
            pass

def path_get_base_name(in_path):
    return os.path.basename(in_path)

def path_get_base_name_without_extension(in_path):
    base_name = os.path.basename(in_path)
    my_array = base_name.split('.')
    return  my_array[0]

def my_close(myset):
    for k in myset:
        k.close()

def my_reader(in_path):
    fid = codecs.open(in_path,'r','utf-8')
    return fid

def my_writer(out_path):
    fid = codecs.open(out_path,'w','utf-8')
    return fid

def write_line(fid, str):
    fid.write(str + "\n")

def file_2_list(input_file):
    fid = codecs.open(input_file,'r','utf-8')
    out_list = []
    for line in fid:
        line = line.strip()
        if not line:
            break
        out_list.append(line)
    fid.close()
    return out_list


def file_2_hash(input_list):
    my_hash = {}
    f = codecs.open(input_list, 'r','utf-8')
    for line in f:
        line = line.strip()
        if not line:
            continue
        my_hash[line] = 1
    f.close()
    return my_hash

def dir_get_file_list(input_dir, my_extension):
    input_list = os.listdir(input_dir)
    out_list = [i for i in input_list if i.endswith(my_extension)]
    return out_list

def dir_get_dir_list(input_dir):
    return os.listdir(input_dir)

def replace_at_start(pattern, replacement, in_str):
    s = re.sub('^' + pattern, replacement, in_str)
    return s

def replace_at_end(pattern, replacement, in_str):
    s = re.sub(pattern + '$', replacement, in_str)
    return s

def str_2_sec(in_str):
    sec = 0.0
    arr =  in_str.split(':')
    if len(arr) == 1:
        sec = float(in_str)
    elif len(arr) == 2:
        sec = float(arr[0])*60 + float(arr[1])
    elif len(arr) == 3:
        sec = float(arr[0])*60*60 + float(arr[1])*60 + float(arr[2])
    return sec

def mono_2_dct(in_mono, out_dct):
    fid_in = codecs.open(in_mono, 'r', 'utf-8')
    fid_out = codecs.open(out_dct, 'w', 'utf-8')
    for line in fid_in:
        line = line.strip()
        if not line:
            break
        write_line(fid_out, line + " " + line)
    fid_in.close()
    fid_out.close()

def to_string(a,n):
    s = "{0:." + str(n) + "f}"
    return s.format(a)

def flat_list(some_list = []):
    elements=[]
    for item in some_list:
        if type(item) == type([]):
            elements += flat_list(item)
        else:
            elements.append(item)
    return elements

def my_round(x):
    return int(x+0.5)

def write_csh_header(w):
    write_line(w,"#!/bin/csh -f")
    write_line(w, "")

def my_write_float_to_file(in_val,out_file_path):
    w = my_writer(out_file_path)
    write_line(w,to_string(in_val,4))
    w.close()

def read_matrix_from_file(in_file):
    r = my_reader(in_file)
    nor = 0
    for line in r:
        line = line.strip()
        if not line:
            continue
        nor += 1
    r.close()

    r = my_reader(in_file)
    header_line = next(r)
    header_line = header_line.strip()
    header_array = header_line.split("\t")
    noc = len(header_array)

    out_mat = [["" for j in range(noc)] for j in range(nor)]
    out_mat[0][:] = header_array

    for i in range(1,nor):
        line = next(r).strip()
        a = line.split("\t")
        out_mat[i][:] = a

    r.close()
    return out_mat

def read_confusion_matrix_from_file(in_file):
    r = my_reader(in_file)
    conf_mat_size = 0
    for line in r:
        line = line.strip()
        if not line:
            continue
        conf_mat_size += 1
    r.close()

    r = my_reader(in_file)
    header_line = next(r)
    header_line = header_line.strip()
    header_array = header_line.split("\t")


    out_mat = [["" for j in range(conf_mat_size)] for j in range(conf_mat_size)]
    out_mat[0][1:] = header_array

    for i in range(1,conf_mat_size):
        line = next(r).strip()
        a = line.split("\t")
        out_mat[i][:] = a

    r.close()
    return out_mat

def write_confusion_matrix(conf_mat_file,value_factor,value_precison,conf_mat_size,id2label,conf_mat):
    w = my_writer(conf_mat_file)
    first_line = "\t"
    for i in range(0, conf_mat_size):
        first_line += id2label[i] + "\t"
    write_line(w, first_line)
    k = 0
    for my_row in conf_mat:
        current_line = id2label[k] + "\t"
        for val in my_row:
            current_line += to_string(val*value_factor,value_precison) + "\t"
        write_line(w, current_line)
        k += 1
    w.close()

# We assume that file is in PCM format with sampling frequency of 16000Hz
def get_wav_duration(in_file):
    return (os.path.getsize(in_file) - 44)/32000.0

# def pearsonr(x, y):
#   # Assume len(x) == len(y)
#   n = len(x)
#   sum_x = float(sum(x))
#   sum_y = float(sum(y))
#   sum_x_sq = sum(map(lambda x: pow(x, 2), x))
#   sum_y_sq = sum(map(lambda x: pow(x, 2), y))
#   psum = sum(imap(lambda x, y: x * y, x, y))
#   num = psum - (sum_x * sum_y/n)
#   den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
#   if den == 0: return 0
#   return num / den


def continuous_replace(src,dest,inString):
    prevString = inString

    while True:
        inString = inString.replace(src,dest)
        if inString == prevString:
            break
        prevString = inString

    return inString

# RETURNS RANDOM INTEGER BETWEEN START AND END INCLUDING START ADN END.
def random_index(start_ind,end_ind):
    return randint(start_ind,end_ind)

def random_float(start_float,end_float):
    return randfloat()


# Compare two lists
def my_list_compare(list1,list2):
    if len(list1) != len(list2):
        return False

    for i in range(0,len(list1)):
        if list1[i] != list2[i]:
            return False

    return True

def write_wav(sampleRate,in_array,out_wav_name):
    import wave, struct, math, random

    wavef = wave.open(out_wav_name, 'w')
    wavef.setnchannels(1)  # mono
    wavef.setsampwidth(2)  # number of bytes per sample
    wavef.setframerate(sampleRate)
    wavef.setnframes(len(in_array))

    max_val = np.max(np.abs(in_array))
    assert max_val <= 1, "ERROR. IT IS ASSUMED THAT MAXIMAL VALUE OF INPUT ARRAY IS LESS THAN 1"

    for i in range(len(in_array)):
        value = int(32767.0 * in_array[i])
        data = struct.pack('<h', value)
        wavef.writeframesraw(data)

    wavef.close()

########################################################################

def write_wav_2(sampleRate,in_array,out_wav_name):
    import scipy.io.wavfile

    in_array = (32767 * in_array).astype('short')

    scipy.io.wavfile.write(out_wav_name, sampleRate, in_array)

########################################################################
