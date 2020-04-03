#!/usr/bin/env python

import re
import glob
import csv
import os
import sys
import pprint
import logging
import copy
import pickle
#import psutil

import argparse
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Personal files #
    
def get_options():
    """
    Parse and return the arguments provided by the user.
    """
    parser = argparse.ArgumentParser(description='MoMEMtaNeuralNet : A tool to regress the Matrix Element Method with a Neural Network')

    # Scan, deploy and restore arguments #
    a = parser.add_argument_group('Scan, deploy and restore arguments')
    a.add_argument('-s','--scan', action='store', required=False, type=str, default='',
        help='Name of the scan to be used (modify scan parameters in NeuralNet.py)')
    a.add_argument('-task','--task', action='store', required=False, type=str, default='',
        help='Name of dict to be used for scan (Used by function itself when submitting jobs or DEBUG)')
    a.add_argument('--generator', action='store_true', required=False, default=False, 
        help='Wether to use a generator for the neural network')
    a.add_argument('--resume', action='store_true', required=False, default=False,
        help='Wether to resume the training of a given model (path in parameters.py)')

    # Splitting and submitting jobs arguments #
    b = parser.add_argument_group('Splitting and submitting jobs arguments')
    b.add_argument('-split','--split', action='store', required=False, type=int, default=0,
        help='Number of parameter sets per jobs to be used for splitted training for slurm submission (if -1, will create a single subdict)')
    b.add_argument('-submit','--submit', action='store', required=False, default='', type=str,
        help='Wether to submit on slurm and name for the save (must have specified --split)')
    b.add_argument('-resubmit','--resubmit', action='store', required=False, default='', type=str,
        help='Wether to resubmit failed jobs given a specific path containing the jobs that succeded')
    b.add_argument('-debug','--debug', action='store_true', required=False, default=False,
        help='Debug mode of the slurm submission, does everything except submit the jobs')

    # Analyzing or producing outputs for given model (csv or zip file) #
    c = parser.add_argument_group('Analyzing or producing outputs for given model (csv or zip file)')
    c.add_argument('-r','--report', action='store', required=False, type=str, default='',
        help='Name of the csv file for the reporting (without .csv)')
    c.add_argument('-m','--model', action='store', required=False, type=str, default='',                                                                                                          
        help='Loads the provided model name (without .zip and type, it will find them)') 
    c.add_argument('--test', action='store_true', required=False, default=False,
        help='Applies the provided model (do not forget -o) on the test set and output the tree') 
    c.add_argument('-o','--output', action='store', required=False, nargs='+', type=str, default=[], 
        help='Applies the provided model (do not forget -o) on the list of keys from sampleList.py (separated by spaces)') 

    # Concatenating csv files arguments #
    d = parser.add_argument_group('Concatenating csv files arguments')
    d.add_argument('-csv','--csv', action='store', required=False, type=str, default='',
        help='Wether to concatenate the csv files from different slurm jobs into a main one, \
              please provide the path to the csv files')

    # Additional arguments #
    e = parser.add_argument_group('Additional arguments')
    e.add_argument('-v','--verbose', action='store_true', required=False, default=False,
        help='Show DEGUG logging')
    e.add_argument('--GPU', action='store_true', required=False, default=False,
        help='GPU requires to execute some commandes before')

    opt = parser.parse_args()

    if opt.split!=0 or opt.submit!='':
        if opt.scan!='' or opt.report!='':
            logging.critical('These parameters cannot be used together')  
            sys.exit(1)
    if opt.submit!='': # Need --output or --split arguments
        if opt.split==0 and len(opt.output)==0:
            logging.warning('In case of learning you forgot to specify --split')
            sys.exit(1)
    if opt.split!=0 and (opt.report!='' or opt.output!='' or opt.csv!='' or opt.scan!=''):
        logging.warning('Since you have specified a split, all the other arguments will be skipped')
    if opt.csv!='' and (opt.report!='' or opt.output!='' or opt.scan!=''):
        logging.warning('Since you have specified a csv concatenation, all the other arguments will be skipped')
    if opt.report!='' and (opt.output!='' or opt.scan!=''):
        logging.warning('Since you have specified a scan report, all the other arguments will be skipped')
    if (opt.test or len(opt.output)!=0) and opt.output == '': 
        logging.critical('You must specify the model with --output')
        sys.exit(1)
    if opt.generator:
        logging.info("Will use the generator")
    if opt.resume:
        logging.info("Will resume the training of the model")

    return opt

def main():
    #############################################################################################
    # Preparation #
    #############################################################################################
    # Get options from user #
    logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%m/%d/%Y %H:%M:%S')
    opt = get_options()
    # Verbose logging #
    if not opt.verbose:
        logging.getLogger().setLevel(logging.INFO)


    # Private modules containing Pyroot #
    from NeuralNet import HyperModel
    from import_tree import LoopOverTrees
    from produce_output import ProduceOutput
    from parameterize_classifier import ParametrizeClassifier
    from make_scaler import MakeScaler
    from submit_on_slurm import submit_on_slurm
    from generate_mask import GenerateMask
    from split_training import DictSplit
    from concatenate_csv import ConcatenateCSV
    from sampleList import samples_dict, samples_path
    from signal_coupling import Decoupler, Repeater
    from threadGPU import utilizationGPU
    import parameters

    # Needed because PyROOT messes with argparse

    logging.info("="*88)
    logging.info("  _____   _    __  __            _     _            _                          _             ")
    logging.info(" |__  /  / \  |  \/  | __ _  ___| |__ (_)_ __   ___| |    ___  __ _ _ __ _ __ (_)_ __   __ _ ")
    logging.info("   / /  / _ \ | |\/| |/ _` |/ __| '_ \| | '_ \ / _ \ |   / _ \/ _` | '__| '_ \| | '_ \ / _` |")
    logging.info("  / /_ / ___ \| |  | | (_| | (__| | | | | | | |  __/ |__|  __/ (_| | |  | | | | | | | | (_| |")
    logging.info(" /____/_/   \_\_|  |_|\__,_|\___|_| |_|_|_| |_|\___|_____\___|\__,_|_|  |_| |_|_|_| |_|\__, |")
    logging.info("                                                                                       |___/ ")
    logging.info("="*88)

    # Make path model #
    path_model = os.path.join(parameters.main_path,'model')
    if not os.path.exists(path_model):
        os.mkdir(path_model)

    #############################################################################################
    # Splitting into sub-dicts and slurm submission #
    #############################################################################################
    if opt.submit != '':
        if opt.split != 0:
            DictSplit(opt.split,opt.submit,opt.resubmit)
            logging.info('Splitting jobs done')
        
        # Arguments to send #
        args = ' ' # Do not forget the spaces after each arg!
        if opt.generator:           args += '--generator '
        if opt.GPU:                 args += '--GPU '
        if opt.resume:              args += '--resume '
        if opt.model!='':           args += '--model '+opt.model+' '
        if len(opt.output)!=0:      args += '--output '+ ' '.join(opt.output)+' '

        if opt.submit!='':
            logging.info('Submitting jobs with args "%s"'%args)
            if opt.resubmit:
                submit_on_slurm(name=opt.submit+'_resubmit',debug=opt.debug,args=args)
            else:
                submit_on_slurm(name=opt.submit,debug=opt.debug,args=args)
        sys.exit()

    #############################################################################################
    # CSV concatenation #
    #############################################################################################
    if opt.csv!='':
        logging.info('Concatenating csv files from : %s'%(opt.csv))
        dict_csv = ConcatenateCSV(opt.csv)
        dict_csv.Concatenate()
        dict_csv.WriteToFile()

        sys.exit()

    #############################################################################################
    # Reporting given scan in csv file #
    #############################################################################################
    if opt.report != '':
        instance = HyperModel(opt.report)
        instance.HyperReport(parameters.eval_criterion)

        sys.exit()

    #############################################################################################
    # Output of given files from given model #
    #############################################################################################
    list_model = []
    if opt.DY                   : list_model.append('DY') 
    if opt.TT                   : list_model.append('TT') 
    if opt.HToZA                : list_model.append('HToZA') 
    if opt.class_global         : list_model.append('class_global') 
    if opt.class_param          : list_model.append('class_param') 
    if opt.binary               : list_model.append('binary') 
    if opt.ME                   : list_model.append('ME') 

    if opt.model != '' and len(opt.output) != 0:
        # Create directory #
        path_output = os.path.join(parameters.path_out,opt.model)
        if not os.path.exists(path_output):
            os.mkdir(path_output)

        # Instantiate #
        inst_out = ProduceOutput(model=os.path.join(parameters.path_model,opt.model),generator=opt.generator)
        # Loop over output keys #
        for key in opt.output:
            # Create subdir #
            path_output_sub = os.path.join(path_output,key+'_weights')
            if not os.path.exists(path_output_sub):
                os.mkdir(path_output_sub)
            try:
                inst_out.OutputNewData(input_dir=samples_path,list_sample=samples_dict[key],path_output=path_output_sub)
            except Exception as e:
                logging.critical('Could not process key "%s" due to "%s"'%(key,e))
        sys.exit()
    #############################################################################################
    # Data Input and preprocessing #
    #############################################################################################
    # Memory Usage #
    #pid = psutil.Process(os.getpid())
    logging.info('Current pid : %d'%os.getpid())

    # Input path #
    logging.info('Starting tree importation')

    # Import variables from parameters.py
    variables = parameters.inputs+parameters.outputs+parameters.other_variables
    list_inputs  = parameters.inputs
    list_outputs = parameters.outputs

    if not opt.generator:
        # Import arrays #
        logging.info('HToZA samples')
        data = LoopOverTrees(input_dir                 = samples_path,
                                   variables                 = variables,
                                   weight                    = parameters.weights,
                                   reweight_to_cross_section = False,
                                   list_sample               = samples_dict['HToZA'],
                                   cut                       = parameters.cut,
                                   tag =                    'HToZA')
        logging.info('HToZA sample size : {}'.format(data_HToZA.shape[0]))
        logging.info('DY samples')
        data_DY = LoopOverTrees(input_dir                   = samples_path,
                                variables                   = variables,
                                weight                      = parameters.weights,
                                #reweight_to_cross_section = True,
                                reweight_to_cross_section   = False,
                                list_sample                 = samples_dict['DY'],
                                cut                         = parameters.cut,
                                tag                         = 'DY')
        logging.info('DY sample size : {}'.format(data_DY.shape[0]))
        logging.info('TT samples')
        data_TT = LoopOverTrees(input_dir                   = samples_path,
                                variables                   = variables,
                                weight                      = parameters.weights,
                                #reweight_to_cross_section = True,
                                reweight_to_cross_section   = False,
                                list_sample                 = samples_dict['TT'],
                                cut                         = parameters.cut,
                                tag                         = 'TT')
        logging.info('TT sample size : {}'.format(data_TT.shape[0]))



        #logging.info('Current memory usage : %0.3f GB'%(pid.memory_info().rss/(1024**3)))

        # Weight equalization #
        if parameters.weights is not None:
            weight_HToZA = data_HToZA[parameters.weights]
            weight_DY = data_DY[parameters.weights]
            weight_TT = data_TT[parameters.weights]
            min_weight = np.min(np.concatenate((weight_HToZA,weight_DY,weight_TT),axis=0))-0.001 # 0.001 to avoid zero weights
            # By rescaling with min_weight, one avoids the negative weights and keep the difference between them
            weight_HToZA -= min_weight
            weight_DY -= min_weight
            weight_TT -= min_weight

            # We need the different types to have the same sumf of weight to equalize training
            weight_HToZA = weight_HToZA/np.sum(weight_HToZA)*10000 
            weight_DY = weight_DY/np.sum(weight_DY)*10000
            weight_TT = weight_TT/np.sum(weight_TT)*10000
        else:
            weight_HToZA = np.ones(data_HToZA.shape[0])
            weight_DY = np.ones(data_DY.shape[0])
            weight_TT = np.ones(data_TT.shape[0])

        # Check sum of weight #
        if np.sum(weight_HToZA) != np.sum(weight_DY) or np.sum(weight_HToZA) != np.sum(weight_TT) or np.sum(weight_TT) != np.sum(weight_DY):
            logging.warning ('Sum of weights different between the samples')
            logging.warning('\tHToZA : '+str(np.sum(weight_HToZA)))
            logging.warning('\tDY : '+str(np.sum(weight_DY)))
            logging.warning('\tTT : '+str(np.sum(weight_TT)))

        data_HToZA['learning_weights'] = pd.Series(weight_HToZA)
        data_DY['learning_weights'] = pd.Series(weight_DY)
        data_TT['learning_weights'] = pd.Series(weight_TT)
        #logging.info('Current memory usage : %0.3f GB'%(pid.memory_info().rss/(1024**3)))

        # Data splitting #
        mask_HToZA = GenerateMask(data_HToZA.shape[0],parameters.suffix+'_HToZA')
        mask_DY = GenerateMask(data_DY.shape[0],parameters.suffix+'_DY')
        mask_TT = GenerateMask(data_TT.shape[0],parameters.suffix+'_TT')
           # Needs to keep the same testing set for the evaluation of model that was selected earlier
        try:
            train_HToZA = data_HToZA[mask_HToZA==True]
            train_DY = data_DY[mask_DY==True]
            train_TT = data_TT[mask_TT==True]
            test_HToZA = data_HToZA[mask_HToZA==False]
            test_DY = data_DY[mask_DY==False]
            test_TT = data_TT[mask_TT==False]
        except ValueError:
            logging.critical("Problem with the mask you imported, has the data changed since it was generated ?")
            sys.exit(1)
            
        #logging.info('Current memory usage : %0.3f GB'%(pid.memory_info().rss/(1024**3)))
        del data_HToZA, data_DY, data_TT

        
        train_all = pd.concat([train_HToZA,train_DY,train_TT],copy=True).reset_index(drop=True)
        del train_HToZA,train_DY,train_TT # Save space
        test_all = pd.concat([test_HToZA,test_DY,test_TT],copy=True).reset_index(drop=True)
        del test_HToZA,test_DY,test_TT
        #logging.info('Current memory usage : %0.3f GB'%(pid.memory_info().rss/(1024**3)))


        # Parametrized case : add the masses as inputs and make the repetition for each mass #
        if opt.HToZA  and opt.scan!='': # We only need the training set for the scan
            # List of variables to decouple #
            list_to_decouple = parameters.outputs
            decoupled_name = 'weight_HToZA'
            # Modify data #
            logging.info("Starting the training set decoupling")
            train_all = Decoupler(train_all,decoupled_name,list_to_decouple)
            logging.info("\tTraining set decoupled : new size = %d"%train_all.shape[0])
            # Update the list of variables #
            list_inputs += ['mH_MEM','mA_MEM']
            # For testing -> Done in produce_output.py

        if opt.class_param and opt.scan!='':
            signal_name = '-log10(weight_HToZA)'
            train_all = ParametrizeClassifier(train_all,name=signal_name)
            list_inputs = ['-log10(weight_DY)','-log10(weight_TT)',signal_name,'mH_gen','mA_gen']

        # Randomize order, we don't want only one type per batch #
        random_train = np.arange(0,train_all.shape[0]) # needed to randomize x,y and w in same fashion
        np.random.shuffle(random_train) # Not need for testing
        train_all = train_all.iloc[random_train]
          
        # Preprocessing #
        # The purpose is to create a scaler object and save it
        # The preprocessing will be implemented in the network with a custom layer
        if opt.scan!='': # If we don't scan we don't need to scale the data
            MakeScaler(train_all,list_inputs) 
           
        # Turns tags into one-hot vector #
        if opt.class_param or opt.class_global:
            # Instantiate #
            label_encoder = LabelEncoder()
            onehot_encoder = OneHotEncoder(sparse=False)
            label_encoder.fit(train_all['tag'])
            # From strings to labels #
            train_integers = label_encoder.transform(train_all['tag']).reshape(-1, 1)
            test_integers = label_encoder.transform(test_all['tag']).reshape(-1, 1)
            # From labels to strings #
            train_onehot = onehot_encoder.fit_transform(train_integers)
            test_onehot = onehot_encoder.fit_transform(test_integers)
            # From arrays to pd DF #
            train_cat = pd.DataFrame(train_onehot,columns=label_encoder.classes_,index=train_all.index)
            test_cat = pd.DataFrame(test_onehot,columns=label_encoder.classes_,index=test_all.index)
            # Add to full #
            train_all = pd.concat([train_all,train_cat],axis=1)
            test_all = pd.concat([test_all,test_cat],axis=1)

        # Turns signal or background into 0 or 1 #
        if opt.binary:
            # Get the booleans #
            train_target = train_all['tag']=='HToZA'
            test_target = test_all['tag']=='HToZA'
            # From booleans to 0 or 1 #
            train_target *= 1 
            test_target *= 1
            # Turn into DF #
            train_binary = pd.DataFrame(train_target,index=train_all.index)
            train_binary.columns = ['Target_signal']
            test_binary = pd.DataFrame(test_target,index=test_all.index)
            test_binary.columns = ['Target_signal']
            # Concat #
            train_all = pd.concat([train_all,train_binary],axis=1)
            test_all = pd.concat([test_all,test_binary],axis=1)

        logging.info("Sample size seen by network : %d"%train_all.shape[0])
        logging.info("Sample size for the output  : %d"%test_all.shape[0])
        #logging.info('Current memory usage : %0.3f GB'%(pid.memory_info().rss/(1024**3)))
    else:
        logging.info('No samples have been imported since you asked for a generator')
        train_all = pd.DataFrame()
        test_all = pd.DataFrame()
        MakeScaler(generator=True, list_inputs=list_inputs)
    #############################################################################################
    # DNN #
    #############################################################################################
    if opt.GPU:
        # Start the GPU monitoring thread #
        thread = utilizationGPU(print_time = 900,
                                print_current = False,
                                time_step=0.01)
        thread.start()

    if opt.scan != '':
        instance = HyperModel(opt.scan)
        instance.HyperScan(data=train_all,
                           list_inputs=list_inputs,
                           list_outputs=list_outputs,
                           task=opt.task,
                           generator=opt.generator,
                           generator_weights=opt.generator_weights,
                           resume=opt.resume)
        instance.HyperDeploy(best='eval_error')

    if opt.GPU:
        # Closing monitor thread #
        thread.stopLoop()
        thread.join()
        
    if opt.model!='': 
        # Make path #
        output_name = "test" 
        path_output = os.path.join(parameters.path_out,opt.model,output_name)
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        # Instance of output class #
        inst_out = ProduceOutput(model=os.path.join(parameters.main_path,'model',opt.model),generator=opt.generator)

        # Use it on test samples #
        if opt.test:
            logging.info('Processing test output sample '.center(80,'*'))
            inst_out.OutputFromTraining(data=test_all,path_output=path_output)
            logging.info('')
             
   
if __name__ == "__main__":
    main()
