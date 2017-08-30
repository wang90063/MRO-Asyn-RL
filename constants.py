# -*- coding: utf-8 -*-

LOCAL_T_MAX = 20# repeat step size
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
CHECKPOINT_DIR = 'checkpoints'
INITIALIZATION_DIR = 'prenet'
LOG_FILE = 'tmp/a3c_log'
LOG_FILE_3GPP = 'tmp/3gpp_log'
LOG_FILE_TEST = 'tmp/test_log'


PLOT_3GPP_HO_DIR ='plot/3gpp/HO'
PLOT_3GPP_rate_DIR = 'plot/3gpp/rate'

PLOT_RL_HO_DIR ='plot/RL/HO'
PLOT_RL_rate_DIR = 'plot/RL/rate'

PLOT_RL_HO_on_DIR='plot/RLonline/HO'
PLOT_RL_rate_on_DIR='plot/RLonline/rate'

PLOT_BAN_HO_DIR ='plot/BAN/HO'
PLOT_BAN_rate_DIR = 'plot/BAN/rate'

INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate

PARALLEL_SIZE = 1  # parallel thread size
ACTION_SIZE = 6 # number of base stations

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 0.01 # entropy regurarlization constant
MAX_TIME_STEP = 150000
GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = False # To use GPU, set True
TTT = 10

MAX_TIME_STEP_TEST = 16000
num_ite = 200


num_train_data = 100000
batch_size = 100
num_test_data = num_train_data/10
slot = 10

B=4

PARALLEL_ONLINE_SIZE=1

MAX_TIME_STEP_online=166000
