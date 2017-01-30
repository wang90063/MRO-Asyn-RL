# -*- coding: utf-8 -*-
import threading
import tensorflow as tf
import signal
import os
import time

from training_thread_3gpp import TrainingThread3GPP

from constants import PARALLEL_SIZE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import LOG_FILE


global_t = 0

stop_requested = False

training_threads = []

for i in range(PARALLEL_SIZE):
    training_thread = TrainingThread3GPP(i, MAX_TIME_STEP)
    training_threads.append(training_thread)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))

init = tf.global_variables_initializer()
sess.run(init)


# summary for tensorboard
score_input = tf.placeholder(tf.int32)
tf.summary.scalar("score", score_input)

summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

wall_t = 0.0

def train_function(parallel_index):
    global global_t
    training_thread = training_threads[parallel_index]
    # set start_time
    start_time = time.time() - wall_t
    training_thread.set_start_time(start_time)

    while True:
        if stop_requested:
            break
        if global_t > MAX_TIME_STEP:
            break

        diff_global_t = training_thread.process(sess, global_t, summary_writer,
                                                summary_op, score_input)
        global_t += diff_global_t


def signal_handler(signal, frame):
    global stop_requested
    print('You pressed Ctrl+C!')
    stop_requested = True


train_threads = []
for i in range(PARALLEL_SIZE):
    train_threads.append(threading.Thread(target=train_function, args=(i,)))

signal.signal(signal.SIGINT, signal_handler)

# set start time
start_time = time.time() - wall_t

for t in train_threads:
    t.start()

print('Press Ctrl+C to stop')
signal.pause()

print('Now saving data. Please wait')

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


