import os
import copy
import sys
import logging
import numpy as np
import pandas as pd
from root_numpy import array2root

from NeuralNet import HyperModel
from import_tree import Tree2Pandas
from signal_coupling import Decoupler, Recoupler
from parameterize_classifier import ParametrizeClassifier
import parameters


class ProduceOutput:
    def __init__(self,model,generator=False,list_inputs=None):
        self.model = model              # Model of the NeuralNet
        self.list_inputs = list_inputs
        self.generator = generator
        self.generator_filepath = None
        if self.list_inputs is None:
            self.list_inputs = copy.deepcopy(parameters.inputs) 

    def OutputFromTraining(self,data,path_output,output_name=None):
        """
            Get the output of the model from the test set
            This is data separated from the training
            If output_name is specified, the whole data will be written in 'output_name'.root
                if not, the tags in the dataframe are used to split into different files with names 'tag'.root
        """
        inputs = data[self.list_inputs].values
        output = None
        columns = []

        # Get Model Output #
        instance = HyperModel(self.model)
        out = instance.HyperRestore(inputs,generator=self.generator,generator_filepath=self.generator_filepath)
        if output is None: # First element of loop
            output = copy.deepcopy(out) # TODO : fix data_generator for last smaller batch
        else:               # append next elements
            output = np.c_[output,out] 

        # From numpy output array to df #
        output_df = pd.DataFrame(output,columns=columns,index=pd.RangeIndex(start=0,stop=output.shape[0]))

        # Make full df #
        full_df = pd.concat([data,output_df],axis=1)
        # Unresolved issue in DataGenerator :
        # Cannot use smaller batches than batch_size to truncate the last elements of array
        # that do not fit in a last batch
        # TODO : fix that
        # In the meantime, also truncate the output array, otherwise will fill nan
        full_df = full_df[:output.shape[0]]

        # Get the unique tags as a list #
        if output_name is None:
            tag_list = list(full_df['tag'].unique())

            # Loop over tags #
            for tag in tag_list:
                tag_df = full_df.loc[full_df['tag']==tag] # We select the rows corresponding to this tag
                tag_df = tag_df.drop('tag',axis=1)

                # From df to numpy array with dtype #
                tag_output = tag_df.to_records(index=False,column_dtypes='float64')
                tag_output.dtype.names = parameters.make_dtype(tag_output.dtype.names)# because ( ) and . are an issue for root_numpy
                tag_output_name = os.path.join(path_output,tag+'.root')

                # Save as root file #
                array2root(tag_output,tag_output_name,mode='recreate')
                logging.info('Output saved as : '+tag_output_name)
        else:
            # From df to numpy array with dtype #
            full_output = full_df.to_records(index=False,column_dtypes='float64')
            full_output.dtype.names = parameters.make_dtype(full_output.dtype.names)# because ( ) and . are an issue for root_numpy
            full_output_name = os.path.join(path_output,output_name)
            array2root(full_output,full_output_name,mode='recreate')
            logging.info('Output saved as : '+full_output_name)
         
    def OutputNewData(self,input_dir,list_sample,path_output,variables=None):
        """
            Given a model, produce the output 
            The Network has never seen this data !
        """
        # Loop over datasets #
        logging.info('Input directory : %s'%input_dir)
        for f in list_sample: 
            name = os.path.basename(f)
            full_path = os.path.join(input_dir,f)
            logging.info('Looking at %s'%f)

            # Get the data #
            if variables is None:
                var = parameters.inputs+parameters.outputs+parameters.other_variables
            else:
                var = copy.deepcopy(variables) # Avoid bug where variables is changed at each new file
            data = Tree2Pandas(input_file=full_path,
                               variables=var,
                               weight=parameters.weights,
                               cut = parameters.cut,
                               reweight_to_cross_section=False)
                
            if data.shape[0]==0:
                logging.info('\tEmpty tree')
                continue # Avoids empty trees
            
            if self.generator:
                self.generator_filepath = full_path

            self.OutputFromTraining(data=data,path_output=path_output,output_name=name)
