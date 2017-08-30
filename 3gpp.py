# -*- coding: utf-8 -*-
import threading
import tensorflow as tf
import numpy as np
import signal
import os
import time

from training_thread_3gpp import TrainingThread3GPP

from constants import MAX_TIME_STEP_TEST
from constants import CHECKPOINT_DIR
from constants import LOG_FILE_3GPP
from constants import PLOT_3GPP_HO_DIR
from constants import PLOT_3GPP_rate_DIR
from constants import PARALLEL_SIZE
from constants import num_ite


sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))

init = tf.global_variables_initializer()
sess.run(init)

# summary for tensorboard
score_input = tf.placeholder(tf.float32)
rate_input = tf.placeholder(tf.float32)
reward_handover_input = tf.placeholder(tf.float32)
tf.summary.scalar("score", score_input)
tf.summary.scalar("rate", rate_input)
tf.summary.scalar("reward_handover", reward_handover_input)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOG_FILE_3GPP, sess.graph)

def train_function(parallel_index, csv_write_ho, csv_write_rate):
    global global_t
    training_thread = training_threads[parallel_index]
    # set start_time
    start_time = time.time() - wall_t
    training_thread.set_start_time(start_time)

    while True:
        if stop_requested:
            break
        if global_t > MAX_TIME_STEP_TEST:
            break

        diff_global_t, csv_write_ho,csv_write_rate  = training_thread.process(sess, csv_write_ho, csv_write_rate,global_t, summary_writer,
                                                           summary_op, score_input, rate_input, reward_handover_input)
        global_t += diff_global_t


def signal_handler(signal, frame):
    global stop_requested
    print('You pressed Ctrl+C!')
    stop_requested = True

for ite in range(1,num_ite):
  csv_write_ho=[]
  csv_write_rate=[]

  global_t = 0

  stop_requested = False

  training_threads = []

  for i in range(1):
      training_thread = TrainingThread3GPP(i, MAX_TIME_STEP_TEST)
      training_threads.append(training_thread)

  wall_t = 0.0


  train_threads = []
  for i in range(1):
      train_threads.append(threading.Thread(target=train_function, args=(i,csv_write_ho, csv_write_rate)))

  signal.signal(signal.SIGINT, signal_handler)

  # set start time
  start_time = time.time() - wall_t

  for t in train_threads:
      t.start()

  # print('Press Ctrl+C to stop')
  # signal.pause()
  #
  # print('Now saving data. Please wait')

  for t in train_threads:
      t.join()

  if not os.path.exists(CHECKPOINT_DIR):
      os.mkdir(CHECKPOINT_DIR)

  # write wall time
  wall_t = time.time() - start_time
  wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
  with open(wall_t_fname, 'w') as f:
      f.write(str(wall_t))
      f.close()

  write_plot_ho = np.asarray(csv_write_ho)
  fname = PLOT_3GPP_HO_DIR + '/'+'HO'+'ite'+'-'+str(ite)+'.csv'
  np.savetxt(fname, write_plot_ho, delimiter=",")

  write_plot_rate = np.asarray(csv_write_rate)
  fname = PLOT_3GPP_rate_DIR + '/'+'HO'+'ite'+'-'+str(ite)+'.csv'
  np.savetxt(fname, write_plot_rate, delimiter=",")

