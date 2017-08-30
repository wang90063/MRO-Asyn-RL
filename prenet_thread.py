import tensorflow as tf
import numpy as np
from system_model import SystemModel
from constants import ACTION_SIZE
from constants import LOCAL_T_MAX
from constants import num_train_data
from constants import batch_size
from constants import num_test_data
from constants import slot
batch_num = num_train_data/batch_size/LOCAL_T_MAX
class prenet_training_thread(object):
    def __init__(self,
                 thread_index,
                 grad_applier,
                 global_network):

        self._thread_index = thread_index

        self.model = SystemModel()

        self.local_network = pretraining_network(self._thread_index, ACTION_SIZE)

        self.local_network.prepare_loss()

        self.var_refs = [v._ref() for v in self.local_network.get_vars()]

        self.gradients = tf.gradients(
            self.local_network.loss, self.var_refs,
            gate_gradients=False,
            aggregation_method=None,
            colocate_gradients_with_ops=False)

        self.apply_gradients = grad_applier.apply_gradients(
            global_network.get_vars(),
            self.gradients)
        #self.var_refs
        self.sync = self.local_network.sync_from(global_network)

    def train(self,sess,actions_mat,states_mat,actions_test_mat,states_test_mat):

        actions_mat_re = np.reshape(actions_mat, num_train_data)

        states_mat_re = np.reshape(states_mat, [num_train_data, 2 * ACTION_SIZE])

        actions_test_mat_re = np.reshape(actions_test_mat, num_test_data)

        states_test_mat_re = np.reshape(states_test_mat, [num_test_data, 2 * ACTION_SIZE])

        a = np.zeros([num_train_data, ACTION_SIZE])
        for n in range(num_train_data):
            a[n, int(actions_mat_re[n])] = 1

        a = np.reshape(a, [num_train_data / LOCAL_T_MAX, LOCAL_T_MAX, ACTION_SIZE])

        accs = []

        for i in range(batch_num):
            sess.run(self.sync)
            _, loss_value = sess.run([self.apply_gradients, self.local_network.loss], feed_dict={
                self.local_network.s: states_mat_re[i * batch_size * LOCAL_T_MAX:(i + 1) * batch_size * LOCAL_T_MAX, :],
                self.local_network.y_acc: a[i * batch_size:(i + 1) * batch_size, :, :]})
            out = self.local_network.readout_pi.eval(session = sess, feed_dict={self.local_network.s: states_test_mat_re})
            out_acc = np.argmax(out, 1)
            acc = np.mean(out_acc == actions_test_mat_re)
            accs.append(acc)
            if self._thread_index == 0:
                 print('Thread',self._thread_index,'loss', loss_value)
                 print(acc)
            if i * LOCAL_T_MAX == num_train_data/slot:
                self.local_network.reset_state()
        with open("predata/acc.txt" + str(self._thread_index), 'w') as f:
            for acc in accs:
                f.write(str(acc) + '\n')
            f.close()


class network(object):
    def __init__(self,
                 thread_index,
                 action_size):
        self._action_size = action_size
        self._thread_index = thread_index

    def get_vars(self):
        raise NotImplementedError()

    def sync_from(self, src_netowrk,name=None):
        src_vars = src_netowrk.get_vars()
        dst_vars = self.get_vars()

        sync_ops = []

        with tf.name_scope(name, "network", []) as name:
           for (src_var, dst_var) in zip(src_vars, dst_vars):
                sync_op = tf.assign(dst_var, src_var)
                sync_ops.append(sync_op)

        return tf.group(*sync_ops, name = name)




class pretraining_network(network):

    def __init__(self,
                 thread_index,
                 action_size):
        network.__init__(self, thread_index,action_size)
        scope_name = "pi_net_" + str(self._thread_index)
        with tf.variable_scope(scope_name) as scope:
            # network weights
            self.W1,self.b1 = self._fc_variable([12, 8])

            # input data
            self.s = tf.placeholder(tf.float32, shape=[None, 2 * ACTION_SIZE])
            # step_size = tf.placeholder(tf.float32, [1])
            # hidden fc layer
            h_fc1 = tf.nn.relu(tf.matmul(self.s, self.W1) + self.b1)

            # lstm cell
            lstm = tf.nn.rnn_cell.BasicLSTMCell(8, state_is_tuple=True)

            # weight for policy output layer
            self.W2,self.b2 = self._fc_variable([8, ACTION_SIZE])

            h_fc1 = tf.reshape(h_fc1, [-1, int(LOCAL_T_MAX), 8])

            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm, h_fc1,
                                                         dtype=tf.float32,
                                                         scope=scope)

            lstm_outputs = tf.reshape(lstm_outputs, [-1, 8])

            # policy (output)
            self.readout_pi = tf.nn.softmax(tf.matmul(lstm_outputs, self.W2) + self.b2)

            scope.reuse_variables()
            self.W_lstm = tf.get_variable("BasicLSTMCell/Linear/Matrix")
            self.b_lstm = tf.get_variable("BasicLSTMCell/Linear/Bias")

            self.reset_state()

    def reset_state(self):
        self.lstm_state_out = tf.nn.rnn_cell.LSTMStateTuple(np.zeros([1, 8]),
                                                            np.zeros([1, 8]))

    def _fc_variable(self, weight_shape):
        input_channels = weight_shape[0]
        output_channels = weight_shape[1]
        d = 1.0 / np.sqrt(input_channels)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d))
        return weight, bias

    def prepare_loss(self):
        self.y_acc = tf.placeholder(tf.float32, shape=[None, LOCAL_T_MAX, self._action_size])
        self.loss_pi = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.readout_pi, labels=self.y_acc))
        self.loss = self.loss_pi

    def get_vars(self):
        return [self.W1, self.b1,
                self.W_lstm, self.b_lstm,
                self.W2, self.b2]




