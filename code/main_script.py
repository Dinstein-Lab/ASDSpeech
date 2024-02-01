# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:03:54 2021

@author: marinamu
"""

from pathlib import Path
import os
from optparse import OptionParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.chdir(Path(__file__).parent.absolute())


from train_and_test_kfold import TrainTestKFold
from tic_toc_class import tic_toc
from commons_functions import load_yaml


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

# main
# =================================================================================================
if __name__ == "__main__":
    tic_toc.tic()
    options = set_parser()
    config_dict = load_yaml(file_pointer=options.config)

    # Load configurations:
    params_config = config_dict.get('params_config', dict())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params_config.get('GPU_id'))  # 0 or 1. on which GPU to run 21.11.2021

    main_class = TrainTestKFold(config_dict, options)
    main_class.run_all()
    print(" ********** IN TOTAL THE PROCESS TOOK = {:.3f} min ********** ".format(tic_toc.toc()/60))

