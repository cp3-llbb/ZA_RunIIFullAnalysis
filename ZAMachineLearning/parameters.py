# Defines the parameters that users might need to change
# Must be included manually in each script

# /!\ Two types of files need to be filled with users parameters 
#           - parameters.py (mort important one)
#           - sampleList.py (on what samples to run)
#           (optionnaly NeuralNet.py for early_stopping etc)
import multiprocessing
from keras.losses import binary_crossentropy, mean_squared_error, logcosh, cosine_proximity
from keras.optimizers import RMSprop, Adam, Nadam, SGD            
from keras.activations import relu, elu, selu, softmax, tanh, sigmoid
from keras.regularizers import l1,l2 

##################################  Path variables ####################################

main_path = '/home/ucl/cp3/fbury/MoMEMtaNeuralNet/'   
path_out = '/nfs/scratch/fynu/fbury/MoMEMta_output/NNOutput/' 
path_model = '/home/ucl/cp3/fbury/MoMEMtaNeuralNet/model'

##############################  Datasets proportion   #################################
# Total must be 1 #
training_ratio = 0.7    # Training set sent to keras (contains training + evaluation)
evaluation_ratio = 0.1  # evaluation set set sent to autom8
output_ratio = 0.2      # Output set for plotting later
assert training_ratio + evaluation_ratio + output_ratio == 1
# Will only be taken into account for the masks generation, ignored after

############################### Slurm parameters ######################################
# For GPU #
partition = 'cp3-gpu'  # Def, cp3 or cp3-gpu
QOS = 'cp3-gpu' # cp3 or normal
time = '5-00:00:00' # days-hh:mm:ss
mem = '60000' # ram in MB
tasks = '20' # Number of threads(as a string)

# For CPU #
#partition = 'cp3'  # Def, cp3 or cp3-gpu
#QOS = 'cp3' # cp3 or normal
#time = '0-02:00:00' # days-hh:mm:ss
#mem = '60000' # ram in MB
#tasks = '20' # Number of threads(as a string)

######################################  Names  ########################################
# Model name (only for scans)
model = 'NeuralNetModel'       # Classic mode
#model = 'NeuralNetGeneratorModel'  # Generator mode
# scaler and mask names #
suffix = 'resolved' 
    # scaler_name -> 'scaler_{suffix}.pkl'  If does not exist will be created 
    # mask_name -> 'mask_{suffix}_{sample}.npy'  If does not exist will be created 

# Training resume #
resume_model = ''  # Will only be used if set in the arguments of the script

################################## Generator part ####################################
# Generator #
path_gen_training = '/home/ucl/cp3/fbury/scratch/MoMEMta_output/ME_TTBar_generator_all/path3' # For training
path_gen_validation = '/home/ucl/cp3/fbury/scratch/MoMEMta_output/ME_TTBar_generator_all/path0' # For val_loss during training
path_gen_evaluation = '/home/ucl/cp3/fbury/scratch/MoMEMta_output/ME_TTBar_generator_all/path1' # for model evaluation
path_gen_output = '/home/ucl/cp3/fbury/scratch/MoMEMta_output/ME_TTBar_generator_all/path2' # for output

workers = 20 # Only for generator part

# Output #
output_batch_size = 1000

##############################  Evaluation criterion   ################################

eval_criterion = "eval_error" # either val_loss or eval_error
    
#################################  Scan dictionary   ##################################
# /!\ Lists must always contain something (even if 0, in a list !), otherwise 0 hyperparameters #
# Classification #
p = { 
    'epochs' : [5],   
    'batch_size' : [512], 
    'lr' : [0.001], 
    'hidden_layers' : [4], 
    'first_neuron' : [300],
    'dropout' : [0],
    'l2' : [0],
    'activation' : [relu],
    'output_activation' : [tanh],
    'optimizer' : [Adam],  
    'loss_function' : [binary_crossentropy] 
}
repetition = 1 # How many times each hyperparameter has to be used 

###################################  Variables   ######################################
cut = None

#weights = 'total_weight'
weights = None

# Input branches (combinations possible just as in ROOT #
inputs = [
         ]
# Output branches #
outputs = [
          ] 

# Other variables you might want to save in the tree #
other_variables = [
                 ]

################################  dtype operation ##############################
# root_numpy does not like some operators very much #
def make_dtype(list_names): 
    list_dtype = [(name.replace('.','_').replace('(','').replace(')','').replace('-','_minus_').replace('*','_times_')) for name in list_names]
    return list_dtype
        



                                



