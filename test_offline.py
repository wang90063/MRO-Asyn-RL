import tensorflow as tf
import numpy as np

from game_ac_network import GameACLSTMNetwork
from system_model import SystemModel


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

global_t = 0

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


for ite in range(num_ite):

    local_t = 0
    csv_write_ho = []
    csv_write_rate = []
    model.intialize_para()
    while True:
        if local_t > MAX_TIME_STEP_TEST:
            break
        else:
            states = []
            actions = []
            rewards = []
            values = []
            episode_reward = 0
            episode_rate = 0
            t = 0
            for i in range(LOCAL_T_MAX):
                pi_, value_ = global_network.run_policy_and_value(sess, model.s_t)
                action = choose_action(pi_)

                states.append(model.s_t)
                actions.append(action)
                values.append(value_)

                model.state_update(actions[i - 1], actions[i])

                # if  (self.thread_index == 0):
                #   print(self.model.rate, self.model.test_rates, self.thread_index)
                # receive game result
                reward = model.reward
                terminal = model.terminal

                episode_reward += reward

                episode_rate += model.rate
                # if  (self.thread_index == 0):
                #         print("Thread",  self.thread_index, "reward", reward, "episode_reward", self.episode_reward, "global_t", global_t, "local_t", self.local_t, "rate", self.model.rate, "handover_reward", self.model.reward_handover)

                # clip reward
                rewards.append(reward)

                t += 1
                # s_t1 -> s_t
                model.update()
                terminal = model.terminal
                if model.terminal:
                    break

            local_t = local_t + t

            print("local_t", local_t, "interval", t, "user", model.users)

            if terminal:
                handover_ratio = model.count_handover_total / (
                    model.count_no_handover + model.count_handover_total + 1)

                record_score(sess, summary_writer, summary_op, score_input,
                             episode_reward / t, rate_input, episode_rate / t, reward_handover_input,
                             handover_ratio, local_t)
                # if local_t % 100 == 0 and local_t != 0:
                model.count_no_handover = 0
                model.count_handover_total = 0
                csv_write_ho.append([local_t, handover_ratio])
                csv_write_rate.append([local_t, episode_rate / t])
                model.init_users()
                episode_rate = 0
                episode_reward = 0
                global_network.reset_state()
                model.init_users()


            else:

                handover_ratio = model.count_handover_total / (
                    model.count_no_handover + model.count_handover_total + 1)

                record_score(sess, summary_writer, summary_op, score_input,
                             episode_reward / t, rate_input, episode_rate / t, reward_handover_input,
                             handover_ratio, local_t)
                # if local_t % 100 == 0 and local_t != 0:
                #     model.count_no_handover = 0
                #     model.count_handover_total = 0
                csv_write_ho.append([local_t, handover_ratio])
                csv_write_rate.append([local_t, episode_rate / t])
                model.init_users()
                episode_rate = 0
                episode_reward = 0
                global_network.reset_state()


            write_plot_ho = np.asarray(csv_write_ho)
            fname = PLOT_RL_HO_DIR + '/' + 'HO' + 'ite' + '-' + str(ite) + '.csv'
            np.savetxt(fname, write_plot_ho, delimiter=",")

            write_plot_rate = np.asarray(csv_write_rate)
            gname = PLOT_RL_rate_DIR + '/' + 'HO' + 'ite' + '-' + str(ite) + '.csv'
            np.savetxt(gname, write_plot_rate, delimiter=",")
