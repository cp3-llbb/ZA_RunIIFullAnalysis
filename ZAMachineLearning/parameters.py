# Defines the parameters that users might need to change
# Must be included manually in each script

# /!\ Two types of files need to be filled with users parameters 
#           - parameters.py (mort important one)
#           - sampleList.py (on what samples to run)
#           (optionnaly NeuralNet.py for early_stopping etc)
import multiprocessing
from keras.losses import binary_crossentropy, mean_squared_error, logcosh, categorical_crossentropy
from keras.optimizers import RMSprop, Adam, Nadam, SGD            
from keras.activations import relu, elu, selu, softmax, tanh, sigmoid, softmax
from keras.regularizers import l1,l2 
import os

##################################  Path variables ####################################

main_path = os.path.abspath(os.path.dirname(__file__))
path_out = os.path.abspath('~/scratch/ZAMachineLearning_output')
path_model = os.path.join(main_path,'model')

##############################  Datasets proportion   #################################
crossvalidation = True

# Classic training #
# -> For crossvalidation == False
training_ratio = 0.7    # Training set sent to keras
evaluation_ratio = 0.1  # Evaluation set sent to autom8
output_ratio = 0.2      # Output set for plotting later
assert training_ratio + evaluation_ratio + output_ratio == 1 
# Will only be taken into account for the masks generation, ignored after

# Cross-validation #
# -> For crossvalidation == True
N_models = 5  # Number of models to train
N_train  = 3  # Number of slices on which to train
N_eval   = 1  # Number of slices on which to evaluate the model
N_apply  = 1  # Number of slices on which the model will be applied for uses in analysis
N_slices = N_train+N_eval+N_apply
splitbranch = 'event' # Will split the training based on "event % N_slices" 
# /!\ Don't forget to add "splitbranch" in other_variables

if N_slices % N_models != 0: # will not work otherwise
    raise RuntimeError("N_slices [{}] % N_models [{}] should be == 0".format(N_slices,N_models))
if N_apply != N_slices/N_models: # Otherwise the same slice can be applied on several networks
    raise RuntimeError("You have asked {} models that should be applied {} times, the number of slices should be {} but is {}+{}+{}".format(N_models,N_apply,N_models*N_apply,N_train,N_eval,N_apply))


############################### Slurm parameters ######################################
# For GPU #
#partition = 'cp3-gpu'  # Def, cp3 or cp3-gpu
#QOS = 'cp3-gpu' # cp3 or normal
#time = '5-00:00:00' # days-hh:mm:ss
#mem = '60000' # ram in MB
#tasks = '20' # Number of threads(as a string)

# For CPU #
partition = 'cp3'  # Def, cp3 or cp3-gpu
QOS = 'cp3' # cp3 or normal
time = '0-08:00:00' # days-hh:mm:ss
mem = '5000' # ram in MB
tasks = '1' # Number of threads(as a string) (not parallel training for classic mode)

######################################  Names  ########################################
# Model name (only for scans)
model = 'NeuralNetModel'       # Classic mode
#model = 'NeuralNetGeneratorModel'  # Generator mode
# scaler and mask names #
suffix = 'resolved_and_boosted' 
    # scaler_name -> 'scaler_{suffix}.pkl'  If does not exist will be created 
    # mask_name -> 'mask_{suffix}_{sample}.npy'  If does not exist will be created 

# Data cache #                                                                                       
train_cache = os.path.join(path_out,'train_cache.pkl' )
test_cache = os.path.join(path_out,'test_cache.pkl' )

# Meta config info #
xsec_json = os.path.join(main_path,'background_{era}_xsec.json')
event_weight_sum_json = os.path.join(main_path,'background_{era}_event_weight_sum.json')

# Training resume #
resume_model = ''

# Output #
output_batch_size = 512
split_name = 'tag' # 'sample' or 'tag' : criterion for output file splitting

##############################  Evaluation criterion   ################################

eval_criterion = "eval_error" # either val_loss or eval_error
    
##############################  Model callbacks ################################
# Early stopping to stop learning after some criterion 
early_stopping_params = {'monitor'   : 'val_loss',  # Value to monitor
                         'min_delta' : 0.,          # Minimum delta to declare an improvement
                         'patience'  : 50,          # How much time to wait for an improvement
                         'verbose'   : 1,           # Verbosity level
                         'mode'      : 'min'}       # Mode : 'auto', 'min', 'max'

# Reduce LR on plateau : if no improvement for some time, will reduce lr by a certain factor
reduceLR_params = {'monitor'    : 'val_loss',   # Value to monitor
                   'factor'     : 0.5,          # Multiplicative factor by which to multiply LR
                   'min_lr'     : 1e-5,         # Minnimum value for LR
                   'patience'   : 10,           # How much time to wait for an improvement
                   'cooldown'   : 5,            # How many epochs before starting again to monitor
                   'verbose'    : 1,            # Verbosity level
                   'mode'      : 'min'}         # Mode : 'auto', 'min', 'max'

#################################  Scan dictionary   ##################################
# /!\ Lists must always contain something (even if 0, in a list !), otherwise 0 hyperparameters #
# Classification #
p = { 
    'epochs' : [200],   
    'batch_size' : [1000], 
    'lr' : [0.01], 
    'hidden_layers' : [2,3,4,5,6], 
    'first_neuron' : [16,32,64,128,256],
    'dropout' : [0,0.25,0.5],
    'l2' : [0,0.1,0.2],
    'activation' : [selu,relu],
    'output_activation' : [softmax],
    'optimizer' : [Adam],  
    'loss_function' : [categorical_crossentropy] 
}
#p = { 
#    'epochs' : [100],   
#    'batch_size' : [512], 
#    'lr' : [0.001], 
#    'hidden_layers' : [3], 
#    'first_neuron' : [64],
#    'dropout' : [0],
#    'l2' : [0],
#    'activation' : [relu],
#    'output_activation' : [softmax],
#    'optimizer' : [Adam],  
#    'loss_function' : [categorical_crossentropy] 
#}

repetition = 1 # How many times each hyperparameter has to be used 

###################################  Variables   ######################################
cut = None

weights = 'total_weight'

# Input branches (combinations possible just as in ROOT #
inputs = [
            'bb_M',
            'llbb_M',
            '$mA',
            '$mH',
         ]
# Output branches #
outputs = [
            '$DY',
            '$TT',
            '$ZA',
          ] 

# Other variables you might want to save in the tree #
other_variables = [
                 ]

################################  dtype operation ##############################
# root_numpy does not like some operators very much #
def make_dtype(list_names): 
    list_dtype = [(name.replace('.','_').replace('(','').replace(')','').replace('-','_minus_').replace('*','_times_')) for name in list_names]
    return list_dtype
        



                                



