
import tensorflow as tf
import numpy as np
from system_model import SystemModel
from constants import ACTION_SIZE
from constants import num_train_data
from constants import batch_size
model = SystemModel()

batch_num = num_train_data/batch_size

actions_mat = np.loadtxt('label.txt')

states_mat = np.loadtxt('data.txt')

actions_mat_re = np.reshape(actions_mat, num_train_data)

states_mat_re = np.reshape(states_mat, [num_train_data, 2*ACTION_SIZE])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def creatnetwork():
  # network weights

  W1 = weight_variable([12,8])
  b1 = bias_variable([8])

  W2 = weight_variable([8 ,ACTION_SIZE])
  b2 = bias_variable([ACTION_SIZE])

  # input layer

  s = tf.placeholder(tf.float32, shape = [None, 12])

  # hidden layers

  h_fc = tf.matmul(s,W1) + b1

  #readout layer

  readout = tf.matmul(h_fc,W2) + b2

  return s,readout

s,readout = creatnetwork()

# loss function
y_acc = tf.placeholder(tf.float32, shape = [None, ACTION_SIZE])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(readout, y_acc))

# train node

# train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
# train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
train_step = tf.train.RMSPropOptimizer(0.01,0.99,0.0,1e-6).minimize(loss)
# out = readout.eval(feed_dict={s: states_mat_re})
a = np.zeros([num_train_data, ACTION_SIZE])
for n in range(num_train_data):
    # print(actions_mat_re[n])
    a[n,int(actions_mat_re[n])] = 1

states_mat_re[0:num_train_data-batch_size-1,:] -= np.mean(states_mat_re[0:num_train_data-batch_size-1,:],axis=0)
states_mat_re[0:num_train_data-batch_size-1,:] /= np.std(states_mat_re[0:num_train_data-batch_size-1,:],axis=0)

accs = []

for i in range(batch_num-1):
    m = np.random.randint(0,num_train_data-batch_size-1,batch_size)
    _,loss_value=sess.run([train_step,loss],feed_dict={s:states_mat_re[m,:],y_acc:a[m,:]})
    print(loss_value)
    # print(a[m,:])
    out = readout.eval(feed_dict={s: states_mat_re[(num_train_data-batch_size):,:]})
    out_acc = np.argmax(out,1)
    # print(out_acc)
    acc = np.mean(out_acc == actions_mat_re[(num_train_data-batch_size):])
    accs.append(acc)
    # print(acc)
with open("acc.txt",'w') as f:
    for acc in accs:
        f.write(str(acc)+'\n')
    f.close()