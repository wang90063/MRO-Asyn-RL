import tensorflow as tf
import numpy as np

from game_ac_network import GameACLSTMNetwork
from system_model import SystemModel
import time

from constants import ACTION_SIZE
from constants import CHECKPOINT_DIR

from constants import MAX_TIME_STEP_TEST
from constants import USE_GPU
from constants import LOCAL_T_MAX
from constants import LOG_FILE_TEST

from constants import PLOT_RL_HO_DIR
from constants import PLOT_RL_rate_DIR
from constants import num_ite


def choose_action( pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

def record_score(sess, summary_writer, summary_op, score_input, score, rate_input, rate, reward_handover_input, reward_handover, global_t):
    summary_str = sess.run(summary_op, feed_dict={
      score_input: score, rate_input:rate, reward_handover_input:reward_handover
    })
    summary_writer.add_summary(summary_str, global_t)#
    summary_writer.flush()


model = SystemModel()



device = "/cpu:0"
if USE_GPU:
  device = "/gpu:0"



global_network = GameACLSTMNetwork(ACTION_SIZE, -1, device)


# prepare session
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
summary_writer = tf.summary.FileWriter(LOG_FILE_TEST, sess.graph)

#load checkpoint with saver
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded:", checkpoint.model_checkpoint_path)
  tokens = checkpoint.model_checkpoint_path.split("-")
  global_t = int(tokens[1])
  print(">>> global step set: ", global_t)
  # set wall time
  wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
  with open(wall_t_fname, 'r') as f:
      wall_t = float(f.read())
else:
    print("Could not find old checkpoint and pretrain")



for ite in range(4):
    wall_t = 0.0

    start_time = time.time() - wall_t
    print(ite)
    episode_count = 0
    local_t = 0
    count = 0
    csv_write_ho = []
    csv_write_rate = []
    model.intialize_para()
    episode_reward = 0
    episode_rate = 0
    states = []
    actions = []
    rewards = []
    values = []
    while True:
        if episode_count == 1:
            break
        else:


            pi_, value_ = global_network.run_policy_and_value(sess, model.s_t)
            action = choose_action(pi_)

            states.append(model.s_t)
            actions.append(action)
            values.append(value_)

            model.state_update(actions[count - 1], actions[count])

            reward = model.reward
            terminal = model.terminal

            episode_reward += reward

            episode_rate += model.rate

            rewards.append(reward)

            count += 1
            # s_t1 -> s_t
            model.update()
            terminal = model.terminal

            # if terminal:
            #     handover_ratio = model.count_handover_total / (
            #         model.count_no_handover + model.count_handover_total + 1)
            #
            #     model.count_no_handover = 0
            #     model.count_handover_total = 0
            #     episode_count += 1
            #     print(count, episode_rate / count, episode_count)
            #     csv_write_ho.append([episode_count, handover_ratio])
            #     csv_write_rate.append([episode_count, episode_rate/count])
            #     model.init_users()
            #     episode_rate = 0
            #     episode_reward = 0
            #     global_network.reset_state()
            #     model.init_users()
            #     count = 0
            #     states = []
            #     actions = []
            #     rewards = []
            #     values = []


            if time.time()-start_time > 2:

                handover_ratio = model.count_handover_total / (
                    model.count_no_handover + model.count_handover_total + 1)

                model.count_no_handover = 0
                model.count_handover_total = 0
                episode_count += 1
                print(count,episode_rate / count, handover_ratio)
                csv_write_ho.append([episode_count, handover_ratio])
                csv_write_rate.append([episode_count, episode_rate / count])
                count = 0
                model.init_users()
                episode_rate = 0
                episode_reward = 0
                global_network.reset_state()
                states = []
                actions = []
                rewards = []
                values = []

    write_plot_ho = np.asarray(csv_write_ho)
    fname = PLOT_RL_HO_DIR + '/' + 'HO' + 'ite'+'-'+str(ite) + '.csv'
    np.savetxt(fname, write_plot_ho, delimiter=',')

    write_plot_rate = np.asarray(csv_write_rate)
    fname = PLOT_RL_rate_DIR + '/' + 'rate' + 'ite'+'-'+str(ite)  + '.csv'
    np.savetxt(fname, write_plot_rate, delimiter=',')