# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import numpy as np

import signal
import random
import math
import os
import time

from game_ac_network import GameACLSTMNetwork
from a3c_training_thread_test import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE

from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP_online
from constants import CHECKPOINT_DIR

from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import INITIALIZATION_DIR

from constants import PLOT_RL_HO_on_DIR
from constants import  PLOT_RL_rate_on_DIR

# from constants import PLOT_RE_HO_DIR
# from constants import PLOT_RE_rate_DIR


def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)

device = "/cpu:0"
if USE_GPU:
    device = "/gpu:0"
global_network = GameACLSTMNetwork(ACTION_SIZE, -1, device)

learning_rate_input = tf.placeholder("float")
grad_applier = RMSPropApplier(learning_rate=learning_rate_input,
                              decay=RMSP_ALPHA,
                              momentum=0.0,
                              epsilon=RMSP_EPSILON,
                              clip_norm=GRAD_NORM_CLIP,
                              device=device)
initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                    INITIAL_ALPHA_HIGH,
                                    INITIAL_ALPHA_LOG_RATE)
training_threads=[]
for i in range(2):
    thread = A3CTrainingThread(i, global_network, initial_learning_rate,
                                      learning_rate_input,
                                      grad_applier, MAX_TIME_STEP_online,
                                      device = device)
    training_threads.append(thread)


for ite in range(431,500):
    csv_write_ho = []
    csv_write_rate = []
    print(ite)
    global_t = 0

    acc_t = 0

    episode_count = 0

    stop_requested = False

    training_thread = training_threads[1]

    # prepare session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                            allow_soft_placement=True))

    init = tf.global_variables_initializer()
    sess.run(init)

    # summary for tensorboard
    score_input = tf.placeholder(tf.float32)
    rate_input = tf.placeholder(tf.float32)
    reward_handover_input = tf.placeholder(tf.float32)
    # tf.summary.scalar("score", score_input)
    # tf.summary.scalar("rate", rate_input)
    # tf.summary.scalar("reward_handover", reward_handover_input)
    # summary_op = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

    # init or load checkpoint with saver
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    pretrain = tf.train.get_checkpoint_state(INITIALIZATION_DIR)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("checkpoint loaded:", checkpoint.model_checkpoint_path)
        tokens = checkpoint.model_checkpoint_path.split("-")
        # set global step
        global_t = int(tokens[1])
        print(">>> global step set: ", global_t)
        # set wall time
        wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
        with open(wall_t_fname, 'r') as f:
            wall_t = float(f.read())
    else:
        print("Could not find old checkpoint and pretrain5")
    wall_t = 0.0

    start_time = time.time() - wall_t
    training_thread.set_start_time(start_time)


    while True:
        # if global_t > MAX_TIME_STEP_online:
        #     break
        if episode_count ==1:
            break

        diff_global_t = training_thread.process(sess, global_t)  # , summary_writer,
        # summary_op, score_input, rate_input, reward_handover_input)
        acc_t += diff_global_t

        # if training_thread.model.terminal:
        #     episode_count += 1
        #     handover_ratio = training_thread.model.count_handover_total / (
        #         training_thread.model.count_no_handover + training_thread.model.count_handover_total + 1)
        #     training_thread.model.count_no_handover = 0
        #     training_thread.model.count_handover_total = 0
        #     training_thread.model.init_users()
        #     training_thread.local_network.reset_state()
        #     episode_rate_ave = training_thread.episode_rate / acc_t
        #     print(acc_t, episode_rate_ave, handover_ratio,episode_count)
        #     acc_t = 0
        #     training_thread.episode_rate = 0
        #     csv_write_ho.append([episode_count, handover_ratio])
        #     csv_write_rate.append([episode_count, episode_rate_ave])
        if time.time()-start_time > 2:
            episode_count += 1
            episode_rate_ave = training_thread.episode_rate / acc_t


            handover_ratio = training_thread.model.count_handover_total / (
                training_thread.model.count_no_handover + training_thread.model.count_handover_total + 1)
            training_thread.model.count_no_handover = 0
            training_thread.model.count_handover_total = 0
            csv_write_ho.append([episode_count, handover_ratio])
            csv_write_rate.append([episode_count, episode_rate_ave])
            training_thread.model.init_users()
            training_thread.local_network.reset_state()
            training_thread.episode_rate = 0
            print(acc_t, episode_rate_ave, handover_ratio, episode_count)
            acc_t = 0

    write_plot_ho = np.asarray(csv_write_ho)
    fname =  PLOT_RL_HO_on_DIR+ '/' + 'HO'+'ite'+'-' + str(ite) + '.csv'#PLOT_RL_HO_on_DIR
    np.savetxt(fname, write_plot_ho,delimiter=',')

    write_plot_rate = np.asarray(csv_write_rate)
    fname =  PLOT_RL_rate_on_DIR+ '/' + 'rate'+'ite'+'-' + str(ite) + '.csv'#PLOT_RL_rate_on_DIR
    np.savetxt(fname, write_plot_rate,delimiter=',')
