import tensorflow as tf
import numpy as np
from constants import ACTION_SIZE
from constants import LOCAL_T_MAX
from constants import num_train_data
from constants import batch_size
from constants import CHECKPOINT_DIR
from constants import INITIALIZATION_DIR
from constants import PARALLEL_SIZE
from rmsprop_applier import RMSPropApplier
from prenet_thread import prenet_training_thread
from prenet_thread import pretraining_network
from adam_applier import AdamApplier
import threading
batch_num = num_train_data/batch_size/LOCAL_T_MAX

global_network = pretraining_network(-1, ACTION_SIZE)

training_threads = []

actions_mat = np.loadtxt('predata/label.txt')  # + str(self._thread_index))

states_mat = np.loadtxt('predata/data.txt')  # + str(self._thread_index))

actions_test_mat = np.loadtxt('predata/label_test.txt')  # + str(self._thread_index))

states_test_mat = np.loadtxt('predata/data_test.txt')  # + str(self._thread_index))

# grad_applier = AdamApplier(learning_rate = 0.0001,
#                               beta1=0.9,
#                               beta2=0.999,
#                               epsilon=1e-8,
#                               clip_norm=40.0)

grad_applier = RMSPropApplier(learning_rate = 0.01,
                              decay = 0.99,
                              momentum = 0.0,
                              epsilon = 1e-8,
                              clip_norm = 40.0)

for i in range(PARALLEL_SIZE):

    training_thread = prenet_training_thread( i,
                 grad_applier,
                 global_network)

    training_threads.append(training_thread)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()


def train_function(parallel_index):
    training_thread = training_threads[parallel_index]

    training_thread.train(sess,actions_mat,states_mat,actions_test_mat,states_test_mat)

train_threads = []
for j in range(PARALLEL_SIZE):
    train_threads.append(threading.Thread(target=train_function, args=(j,)))

for t in train_threads:
    t.start()
for t in train_threads:
    t.join()
saver.save(sess, INITIALIZATION_DIR + '/' + 'checkpoint'+'pretrained_net')
print('The pretrained network is saved in file:%s' % CHECKPOINT_DIR)







