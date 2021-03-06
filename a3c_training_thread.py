# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
# from system_model_dif import SystemModel_dif
from system_model import SystemModel
# from system_model_wrong import SystemModel_wrong
from game_ac_network import GameACLSTMNetwork
from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import ACTION_SIZE
from constants import MAX_TIME_STEP
LOG_INTERVAL = 1000
PERFORMANCE_LOG_INTERVAL = 1000

class A3CTrainingThread(object):
  def __init__(self,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device,
               ):
    print ("This is the thread", thread_index)
    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step

    self.local_network = GameACLSTMNetwork(ACTION_SIZE, thread_index, device)

    self.local_network.prepare_loss(ENTROPY_BETA)

    self.actions_thread = []
    with tf.device(device):
      var_refs = [v._ref() for v in self.local_network.get_vars()]
      self.gradients = tf.gradients(
        self.local_network.total_loss, var_refs,
        gate_gradients=False,
        aggregation_method=None,
        colocate_gradients_with_ops=False)

    self.apply_gradients = grad_applier.apply_gradients(
      global_network.get_vars(),
      self.gradients )
      
    self.sync = self.local_network.sync_from(global_network)

    # if self.thread_index >=2 :
    #    self.model = SystemModel_dif()
    # # # elif self.thread_index == 5 :
    # # #    self.model = SystemModel_wrong()
    # # else:
    self.model = SystemModel()
    
    self.local_t = 0

    self.episode_rate = 0

    self.episode_rate_ave = 0

    self.episode_count_local = 0

    self.initial_learning_rate = initial_learning_rate

    self.episode_reward = 0

    # variable controling log output
    self.prev_local_t = 0
    # get the initial state from the initial action
    # self.model.intialize_para()
  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate

  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

  def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):# rate_input, rate, reward_handover_input, reward_handover,
    summary_str = sess.run(summary_op, feed_dict={
      score_input: score})
    # , rate_input:rate, reward_handover_input:reward_handover
    summary_writer.add_summary(summary_str, global_t)#
    summary_writer.flush()
    
  def set_start_time(self, start_time):
    self.start_time = start_time

  def process(self, sess, global_t, summary_writer, summary_op, score_input):#, rate_input,  reward_handover_input):
    states = []
    actions = []
    rewards = []
    values = []

    terminal_end = False

    # copy weights from shared to local
    sess.run( self.sync )

    start_local_t = self.local_t

    start_episode_count = self.episode_count_local

    start_lstm_state = self.local_network.lstm_state_out

    t = 1
    for i in range(LOCAL_T_MAX):
      pi_, value_ = self.local_network.run_policy_and_value(sess, self.model.s_t)
      action = self.choose_action(pi_)

      states.append(self.model.s_t)
      actions.append(action)
      values.append(value_)
      # self.actions_thread.append(action)
      if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):#
        print("pi={}".format(pi_))
        print(" V={}".format(value_))
        print("thread", self.thread_index)
      # print("user", self.model.users)
      # process game

      self.model.state_update(actions[i - 1], actions[i])

      # if  (self.thread_index == 0):
      #   print(self.model.rate, self.model.test_rates, self.thread_index)
      #receive game result
      reward = self.model.reward


      self.episode_reward += reward

      self.episode_rate += self.model.rate
      # if  (self.thread_index == 0):
      #         print("Thread",  self.thread_index, "reward", reward, "episode_reward", self.episode_reward, "global_t", global_t, "local_t", self.local_t, "rate", self.model.rate, "handover_reward", self.model.reward_handover)


      rewards.append(reward)

      # print ('rewards', rewards)
      self.local_t += 1
      t += 1
      # s_t1 -> s_t
      self.model.update()
      # if self.local_t % 500 == 0 and self.local_t != 0:
      # print("score={}".format(self.episode_reward))
      terminal = self.model.terminal
      if self.model.terminal:
          break
    #self.episode_reward / t
    if terminal:
       terminal_end = True
       # self.episode_count_local +=1
       # self.handover_ratio = self.model.count_handover_total / (
       #   self.model.count_no_handover + self.model.count_handover_total + 1)
       # self.episode_rate_ave = self.episode_rate / t
       # self.episode_reward_ave = self.episode_reward / t
       self._record_score(sess, summary_writer, summary_op, score_input,
                          value_, global_t)\
         # , rate_input, self.episode_rate_ave, reward_handover_input,
         #                  self.handover_ratio) #value_
       # if self.local_t % 100 == 0 and self.local_t != 0:
       # self.model.count_no_handover = 0
       # self.model.count_handover_total = 0
       # self.episode_rate = 0
       # self.episode_reward = 0
       # self.local_network.reset_state()
       self.model.init_users()

    else:

       # self.handover_ratio = self.model.count_handover_total / (
       # self.model.count_no_handover + self.model.count_handover_total +1)
       # self.episode_rate_ave = self.episode_rate / t
       # self.episode_reward_ave = self.episode_reward / t
       self._record_score(sess, summary_writer, summary_op, score_input,
                          value_, global_t)
                          # , rate_input,self.episode_rate_ave, reward_handover_input,
                          # self.handover_ratio)
       # if self.local_t %100 == 0 and self.local_t !=0:
       # self.model.count_no_handover = 0
       # self.model.count_handover_total = 0
       # self.episode_rate = 0
       # self.episode_reward = 0
       # self.local_network.reset_state()
       # self.model.init_users()
    R = 0.0
    if not terminal_end:
      R = self.local_network.run_value(sess, self.model.s_t)

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    batch_si = []
    batch_a = []
    batch_td = []
    batch_R = []

    # compute and accmulate gradients
    for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R = ri + GAMMA * R
      td = R - Vi
      a = np.zeros([ACTION_SIZE])
      a[ai] = 1

      batch_si.append(si)
      batch_a.append(a)
      batch_td.append(td)
      batch_R.append(R)

    cur_learning_rate = self._anneal_learning_rate(global_t)

    batch_si.reverse()
    batch_a.reverse()
    batch_td.reverse()
    batch_R.reverse()

    _,loss_value = sess.run( [self.apply_gradients,self.local_network.total_loss],
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.a: batch_a,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,
                  self.local_network.initial_lstm_state: start_lstm_state,
                  self.local_network.step_size : [len(batch_a)],
                  self.learning_rate_input: cur_learning_rate } )

    # _, loss_value = sess.run([self.apply_gradients,self.local_network.total_loss],
    #          feed_dict={
    #              self.local_network.s: batch_si,
    #              self.local_network.a: batch_a,
    #              self.local_network.td: batch_td,
    #              self.local_network.r: batch_R,
    #              self.learning_rate_input: cur_learning_rate})

    # if ((self.thread_index == 0) ):
    #     print(self.model.users)
    if ((self.thread_index == 0) ) and(self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL): #
      self.prev_local_t += PERFORMANCE_LOG_INTERVAL
      elapsed_time = time.time() - self.start_time
      steps_per_sec = global_t / elapsed_time
      print("### Performance :{} thread {} loss {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
          self.thread_index,loss_value, global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t
    diff_episode_count = self.episode_count_local - start_episode_count
    return diff_local_t