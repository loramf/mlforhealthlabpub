'''
Title: 
Authors: Lora Frayling
Last Updated Date: January 15th 2020
Code Author: Ioana Bica (ioana.bica95@gmail.com)
'''


import os
import argparse
import logging

#from CRN_encoder_evaluate import test_CRN_encoder
#from CRN_decoder_evaluate import test_CRN_decoder
from utils.cancer_simulation import get_cancer_sim_data as get_cancer_sim_data_one_hot
from utils.cancer_simulation_dosage import get_cancer_sim_data as get_cancer_sim_data_dosage

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default='C:\\Users\\lora.frayling\\Documents\\PhD Application\\mlforhealthlabpub-2\\alg\\counterfactual_recurrent_network\\models')
    return parser.parse_args()

if __name__ == '__main__':

    #We want to generate 8 datasets to start:
    #Original with CT=RT=2
    #Dosage with CT=RT=0 and CT_dosage=RT_dosage=5 for unrounded
    #Dosage with CT=RT=0 and CT_dosage=RT_dosage=0,2,4,6,8,10 for unrounded

    args = init_arg()

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    #uncomment after running
    pickle_map = get_cancer_sim_data_one_hot(chemo_coeff=2, radio_coeff=2, b_load=False,
                                          b_save=True, model_root=args.results_dir)
        

    for dosage_coeff in range(0, 12, 2):

        pickle_map = get_cancer_sim_data_dosage(chemo_coeff=5, radio_coeff=5, chemo_dosage_coeff=dosage_coeff, 
                                                radio_dosage_coeff=dosage_coeff, b_load=False,
                                                b_save=True, model_root=args.results_dir, round=False)
                        

    
