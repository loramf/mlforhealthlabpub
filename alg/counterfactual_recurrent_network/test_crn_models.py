'''
Title: Estimating counterfactual treatment outcomes over time through adversarially balanced representations
Authors: Ioana Bica, Ahmed M. Alaa, James Jordon, Mihaela van der Schaar
International Conference on Learning Representations (ICLR) 2020

Last Updated Date: January 15th 2020
Code Author: Ioana Bica (ioana.bica95@gmail.com)
'''


import os
import argparse
import logging

from CRN_encoder_evaluate import test_CRN_encoder
from CRN_decoder_evaluate import test_CRN_decoder
from utils.cancer_simulation import get_cancer_sim_data as get_cancer_sim_data_one_hot
from utils.cancer_simulation_dosage import get_cancer_sim_data as get_cancer_sim_data_dosage

import numpy as np
import tensorflow as tf
###
import logging
import pickle
import numpy as np
import os

from utils.evaluation_utils import get_processed_data, get_mse_at_follow_up_time, \
    load_trained_model, write_results_to_file
from CRN_model import CRN_Model
from CRN_decoder_evaluate import *


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default='C:\\Users\\lora.frayling\\Documents\\PhD Application\\mlforhealthlabpub-2\\alg\\counterfactual_recurrent_network\\models')
    parser.add_argument("--model_name", default="crn_test_dosage")
    return parser.parse_args()
#00:38:27.360686:
# 00:48
#1:12
#1:27
#1:47
# #finished 1:55
#
#0.9329644768412115
#3.083007178034157
if __name__ == '__main__':

    tf.set_random_seed(123)
    np.random.seed(123)

    args = init_arg()
    import time
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    train_decoder = False

    t0 = time.time()
    for _ in range(1):
        for dosage_coeff in [0,2,4,6,8,10]:
            print("""
            #####################################################################################################
            Training for dosage coefficient: """ + str(dosage_coeff))
            print("##############################################################################################")
            pickle_map = get_cancer_sim_data_dosage(chemo_coeff=5, radio_coeff=5, chemo_dosage_coeff=dosage_coeff, 
                                        radio_dosage_coeff=dosage_coeff, b_load=True,
                                        b_save=False, model_root=args.results_dir, round=True)
            
            t1 = time.time()
            for treatment_format in ['multi_one_hot','continuous','p_continuous']:
                print("Treatment format: " + treatment_format)
                print("""
                #####################################################################################################
                Treatment format: : """ + treatment_format)
                print("##############################################################################################")
                encoder_model_name = 'encoder_' + args.model_name + "_{}".format(dosage_coeff)
                encoder_hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, encoder_model_name)

                models_dir = '{}/crn_models'.format(args.results_dir)
                if not os.path.exists(models_dir):
                    os.mkdir(models_dir)

                rmse_encoder = test_CRN_encoder(pickle_map=pickle_map, models_dir=models_dir,
                                                encoder_model_name=encoder_model_name,
                                                encoder_hyperparams_file=encoder_hyperparams_file,
                                                b_encoder_hyperparm_tuning=False,
                                                treatment_format=treatment_format,
                                                )


                if train_decoder:
                    decoder_model_name = 'decoder_' + args.model_name + "_{}".format(dosage_coeff)
                    decoder_hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, decoder_model_name)

                    """
                    The counterfactual test data for a sequence of treatments in the future was simulated for a 
                    projection horizon of 5 timesteps. 
                
                    """

                    max_projection_horizon = 5
                    projection_horizon = 5
                    
                    rmse_decoder = test_CRN_decoder(pickle_map=pickle_map, max_projection_horizon=max_projection_horizon,
                                                    projection_horizon=projection_horizon,
                                                    models_dir=models_dir,
                                                    encoder_model_name=encoder_model_name,
                                                    encoder_hyperparams_file=encoder_hyperparams_file,
                                                    decoder_model_name=decoder_model_name,
                                                    decoder_hyperparams_file=decoder_hyperparams_file,
                                                    b_decoder_hyperparm_tuning=False,
                                                    treatment_format=treatment_format)

                logging.info("Chemo coeff {} | Radio coeff {}".format(dosage_coeff, dosage_coeff))
                print("RMSE for one-step-ahead prediction.")
                print(rmse_encoder)

                if train_decoder:
                    print("Results for 5-step-ahead prediction.")
                    print(rmse_decoder)
                t2 = time.time()
                print((t2-t1)/60)
                t1 = t2
                

    tn = time.time()
    print((tn-t0)/60)
