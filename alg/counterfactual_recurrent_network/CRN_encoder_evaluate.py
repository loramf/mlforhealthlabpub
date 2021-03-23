'''
Title: Estimating counterfactual treatment outcomes over time through adversarially balanced representations
Authors: Ioana Bica, Ahmed M. Alaa, James Jordon, Mihaela van der Schaar
International Conference on Learning Representations (ICLR) 2020

Last Updated Date: January 15th 2020
Code Author: Ioana Bica (ioana.bica95@gmail.com)
'''

import logging
import numpy as np
import os

from CRN_model import CRN_Model
from utils.evaluation_utils import write_results_to_file, load_trained_model, get_processed_data



def fit_CRN_encoder(dataset_train, dataset_val, model_name, model_dir, hyperparams_file,
                    b_hyperparam_opt, treatment_format='one_hot'):
    _, length, num_covariates = dataset_train['current_covariates'].shape
    #num_treatments is the treatment shape rather than number of treatments
    num_treatments = dataset_train['current_treatments'].shape[-1]
    num_outputs = dataset_train['outputs'].shape[-1]
    num_inputs = dataset_train['current_covariates'].shape[-1] + dataset_train['current_treatments'].shape[-1]

    params = {'num_treatments': num_treatments,
              'num_covariates': num_covariates,
              'num_outputs': num_outputs,
              'max_sequence_length': length,
              'num_epochs': 100}

    hyperparams = dict()
    num_simulations = 50
    best_validation_mse = 1000000

    if b_hyperparam_opt:
        logging.info("Performing hyperparameter optimization")
        for simulation in range(num_simulations):
            logging.info("Simulation {} out of {}".format(simulation + 1, num_simulations))

            hyperparams['rnn_hidden_units'] = int(np.random.choice([0.5, 1.0, 2.0, 3.0, 4.0]) * num_inputs)
            hyperparams['br_size'] = int(np.random.choice([0.5, 1.0, 2.0, 3.0, 4.0]) * num_inputs)
            hyperparams['fc_hidden_units'] = int(np.random.choice([0.5, 1.0, 2.0, 3.0, 4.0]) * (hyperparams['br_size']))
            hyperparams['learning_rate'] = np.random.choice([0.01, 0.001])
            hyperparams['batch_size'] = np.random.choice([64, 128, 256])
            hyperparams['rnn_keep_prob'] = np.random.choice([0.7, 0.8, 0.9])

            logging.info("Current hyperparams used for training \n {}".format(hyperparams))
            model = CRN_Model(params, hyperparams, treatment_format=treatment_format)
            model.train(dataset_train, dataset_val, model_name + "_" + treatment_format, model_dir)
            validation_mse, _ = model.evaluate_predictions(dataset_val)

            if (validation_mse < best_validation_mse):
                logging.info(
                    "Updating best validation loss | Previous best validation loss: {} | Current best validation loss: {}".format(
                        best_validation_mse, validation_mse))
                best_validation_mse = validation_mse
                best_hyperparams = hyperparams.copy()

            logging.info("Best hyperparams: \n {}".format(best_hyperparams))

        write_results_to_file(hyperparams_file, best_hyperparams)

    else:
        logging.info("Using default hyperparameters")
        best_hyperparams = {
            'rnn_hidden_units': 12,
            'br_size': 18,
            'fc_hidden_units': 18,
            'learning_rate': 0.001,
            'batch_size': 64,
            'rnn_keep_prob': 0.9}
        logging.info("Best hyperparams: \n {}".format(best_hyperparams))
        write_results_to_file(hyperparams_file, best_hyperparams)

    model = CRN_Model(params, best_hyperparams, treatment_format)
    model.train(dataset_train, dataset_val, model_name + "_" + treatment_format, model_dir)



def test_CRN_encoder(pickle_map, models_dir,
                     encoder_model_name, encoder_hyperparams_file,
                     b_encoder_hyperparm_tuning, treatment_format='one_hot'):

    training_data = pickle_map['training_data']
    validation_data = pickle_map['validation_data']
    test_data = pickle_map['test_data']
    scaling_data = pickle_map['scaling_data']

    training_processed = get_processed_data(training_data, scaling_data, treatment_format)
    validation_processed = get_processed_data(validation_data, scaling_data, treatment_format)
    test_processed = get_processed_data(test_data, scaling_data, treatment_format)

    fit_CRN_encoder(dataset_train=training_processed, dataset_val=validation_processed,
                    model_name=encoder_model_name, model_dir=models_dir,
                    hyperparams_file=encoder_hyperparams_file, b_hyperparam_opt=b_encoder_hyperparm_tuning,
                    treatment_format=treatment_format)

    CRN_encoder = load_trained_model(validation_processed, encoder_hyperparams_file, encoder_model_name + "_" + treatment_format, models_dir, treatment_format=treatment_format)
    mean_mse, mse, current_treatments, unscaled_predictions, unscaled_outputs, active_entries = CRN_encoder.evaluate_predictions(test_processed)

    rmse = (np.sqrt(np.mean(mse))) / 1150 * 100  # Max tumour volume = 1150
    if treatment_format == 'multi_one_hot':
        current_treatments = None
    results_dict = {
        'rmse': rmse,
        'mse': mse,
        'current_treatments': current_treatments,
        'predictions': unscaled_predictions, 
        'outputs': unscaled_outputs, 
        'active_entries': active_entries
    }

    write_results_to_file(os.path.join(models_dir, encoder_model_name + "_" + treatment_format + "_test_results.pickle"), results_dict)

    return rmse
