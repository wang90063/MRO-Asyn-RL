
import sys
import random
import numpy as np

from constants import LOCAL_T_MAX
from collections import Counter
from math import exp
from constants import direction
from constants import  direction_index
# 400 pixels represent the largest distance of the area, i.e. 100m
WINDOW_WIDTH = 400 *40 # size of window's width in pixels
WINDOW_HEIGHT = 400 *40# size of windows' height in pixels
cell_radius = 4 # "m" this is the radius of cell
resolution  = cell_radius * 4./WINDOW_WIDTH # meter/pixel, the longest distace in the simulation system is "cell_radius * 4"

outlayer_userrange_x_low = -30 / resolution
outlayer_userrange_x_high = 30 / resolution

outlayer_userrange_y_low = -30 / resolution
outlayer_userrange_y_high = 30 / resolution

outer_radius = 50/ resolution
inner_radius = 22.5 / resolution
Num_CELL = 6
NUM_USER = 1 # In asynchronous deep Q learning, only one user in one thread


# generate cell list, every cell in the lists consists of 5 params:
# (1)loc_x(left) (2)loc_y(top) (3)cell_id
# :return: cell_list: (1)loc_x(left) (2)loc_y(top) (3)cell_id

# cell location
cell_id = np.arange(Num_CELL)

# the locations of cells are fixed and the coordinates are given
# cell_x = [200, 200, 370, 370, 200, 30, 30]
# cell_y = [200, 0, 100, 300, 400, 300, 100]

cell_x = [ 0.0, 170.0*40, 170.0*40, 0.0, -170*40, -170*40]
cell_y = [ 200.0*40, 100.0*40, -100.0*40, -200.0*40, -100*40, 100*40]


cells = np.vstack((cell_x, cell_y,cell_id)).T


# "action" is the "cell_id" selected by the agent
class SystemModel:
  def __init__(self):
      self.init_users()
      self.s_t = self._get_state()
      self.handover_indicator = 0  #np.zeros(LOCAL_T_MAX)
      self.reward_handover = 0
      self.handover_consumption =7.5
      self.terminal = False
      self.direction = direction
      self.direction_index = direction_index
  # def intialize_para(self):
  #     self.count_no_handover = 0
  #     self.count_no_failure = 0
  #     self.count_failure = 0
  #     self.count_handover_total=0
  #     self.count_handover = 0


  def state_update(self, last_action, action):
      """
      the func can generate the reward and state, also update the users locations
      :param: users:the locations of the users
              action: the "cell_id" selected by users
              last_action
      """
      self.rates = get_rate_percell(self.users, cells)
      s_t = self._get_state()
      r, rate = self._get_reward(last_action,action)
      self.test_rates=self.rates
      self._move_user()
      self.rates = get_rate_percell(self.users, cells)
      s_t1 = self._get_state()
      self.reward = r
      self.s_t = s_t
      self.s_t1 = s_t1
      self.rate = rate


  def update(self):
      self.s_t = self.s_t1


  def _move_user(self):
      """
      low mobility users are considered, i.e. the user only move one pixel every frame. different mobility trajectories will be tested to present the robustness of the neural network
      """
      self.terminal = False
      mobility_speed = 1  # np.random.randint(1, 3)
      index = np.random.choice(self.direction_index, p=[0.25, 0.25, 0.25, 0.25])
      self.users = self.users + self.direction[index] * mobility_speed

      if np.abs(self.users[0]) > WINDOW_WIDTH/2or np.abs(self.users[1]) > WINDOW_WIDTH/2:#np.abs(self.users[0]) > np.sqrt(3)/2.0 * outer_radius or (np.abs(self.users[0])+np.abs(self.users[1])/np.sqrt(3)) > outer_radius: #and (np.abs(user_x_tmp) > np.sqrt(3)/2.0*inner_radius or (np.abs(user_x_tmp)+np.abs(user_y_tmp)/np.sqrt(3)) > inner_radius):
          self.terminal = True


  def _get_reward(self, last_action, action):
      """
      :param users: the location of user before moving
      :param action: the taken action to obtain "users"
      :return: reward : the weighted sum of rate and reward for handover, i.e. "handover error occurs" -- 0, "handover successes" -- 1
      """
      # reward_weight_rate =0.5



      rate = self.rates[action]
      # last_rate = self.rates[last_action]

      if action == last_action:
         self.handover_indicator = 0
         # self.count_no_handover += 1.0
      else:
         self.handover_indicator = 1
         # self.count_handover_total += 1.0

      reward = rate -  self.handover_indicator * self.handover_consumption
      # if action == last_action and self.count_handover < LOCAL_T_MAX :#andrate >0.5
      #      self.reward_handover = 0.5
      #      self.handover_indicator[self.count_handover] = 0
      #      self.count_no_handover += 1.0
      #      self.count_handover +=1.0
      # # if action == self.serve_cell_id and rate <0.5:
      # #     self.reward_handover = 0
      # #     self.handover_indicator[self.count_handover] = 0
      # #     self.count_handover += 1
      # elif (self.handover_indicator[0] == 1 ):#or rate<0.5
      #      self.reward_handover = -1
      #      self.count_handover_total += 1.0
      #      self.count_handover = 0
      #      self.handover_indicator[self.count_handover] = 1
      # elif  self.handover_indicator[0] != 1 and rate > last_rate:#and rate>0.5
      #      self.reward_handover = 1
      #      self.count_handover = 0
      #      self.handover_indicator[self.count_handover] = 1
      #      self.count_handover_total += 1.0
      # elif self.handover_indicator[0] != 1 and rate < last_rate:
      #     self.reward_handover = -1
      #     self.count_handover = 0
      #     self.handover_indicator[self.count_handover] = 1
      #     self.count_handover_total += 1.0





      # self.reward_rate = rate - last_rate
      # self.reward_rate = np.clip(differ_rate,-5,5)

      # if action != last_action:
      #     self.count_handover += 1.0
      # else:
      #     self.count_no_handover += 1.0
      #
      # if rate < last_rate:
      #
      #     self.reward_rate = -0.1
      #
      # else:
      #     self.reward_rate = 0.1

      self.serve_cell_id = action






      # if rate > 1.5:
      #     self.count_no_failure +=1.0
      # else:
      #     self.count_failure += 1.0



      # if self.count_failure + self.count_no_failure == LOCAL_T_MAX:
      #     self.rate_fail_ratio = self.count_failure / (self.count_failure + self.count_no_failure)
      #     self.count_failure = 0.0
      #     self.count_no_failure = 0.0

      # if rate > last_rate:
      #     reward_rate = 1
      # elif rate == last_rate:
      #     reward_rate = 0
      # elif rate < last_rate:
      #     reward_rate = -1

      # if self.count_handover + self.count_no_handover < 1000:
      #     self.reward_handover = 0
      # elif self.count_handover + self.count_no_handover == 1000:
      #
      #     if self.count_handover/(self.count_no_handover+self.count_handover) > 0.01:
      #
      #         self.reward_handover = -1.0
      #     else:
      #         self.reward_handover = 1.0


      # if self.count_failure + self.count_no_failure < LOCAL_T_MAX:
      #     reward_rate = 0
      # elif self.count_failure + self.count_no_failure == LOCAL_T_MAX:
      #
      #     if self.count_failure / (self.count_failure + self.count_no_failure) > 0.1:
      #         reward_rate = -10
      #     else:
      #         reward_rate = 10
      #     self.count_failure = 0.0
      #     self.count_no_failure = 0.0


      # print (reward_rate, self.reward_handover)
      # diff_rate = 10*(rate-last_rate)
      # reward = self.reward_handover
      return reward, rate

  def _get_state(self):
      feature_serve_vector = np.zeros(Num_CELL)
      feature_serve_vector[self.serve_cell_id] = 1
      # feature_handover = np.zeros(2)
      # if action == self.last_serve_cell_id:  #self.reward_handover == 1:
      #     feature_handover[0] = 1
      # else:
      #     feature_handover[1] = 1
      # s_t = self.rates
      s_t = np.hstack((self.rates,feature_serve_vector))#feature_user_rates_vector,feature_handover

      # s_t = feature_user_rates_vector
      return s_t

  def init_users(self):
    """
    initialize user. every user consists of 4 params:
    (1) loc_x(center) (2) loc_y(center) (3) which cell user is in (4) user mobility type
    user mobility type is divided into 3 categories: low, medium and high. Low mobility users takes 70% of all,
    while medium and high takes 20% and 10%.
    :return: user: (1) loc_x(center) (2) loc_y(center) (3) which cell user is in (4) user mobility type
    """
    while True:
        user_x_tmp = np.random.randint(-40, -20, size=NUM_USER, dtype='int')
        user_y_tmp = np.random.randint(-40, -20, size=NUM_USER, dtype='int')
        if np.abs(user_x_tmp)<WINDOW_WIDTH/2 and np.abs(user_y_tmp)<WINDOW_WIDTH/2:#np.abs(user_x_tmp) < np.sqrt(3)/2.0 * outer_radius and (np.abs(user_x_tmp)+np.abs(user_y_tmp)/np.sqrt(3)) < outer_radius: #and (np.abs(user_x_tmp) > np.sqrt(3)/2.0*inner_radius or (np.abs(user_x_tmp)+np.abs(user_y_tmp)/np.sqrt(3)) > inner_radius):
           user_x = user_x_tmp
           user_y = user_y_tmp
           break
    self.users = np.hstack((user_x, user_y))
    self.rates = get_rate_percell(self.users, cells)
    self.serve_cell_id= np.argmax(self.rates)
    self.last_serve_cell_id = np.random.randint(Num_CELL)

def get_rate_percell(users, cells):
      """
      get the rates of the user in all the cells if this user connects to the cell. return the array "rate" to represent the rate in the cells
      """
      channels_square = np.random.normal(0,1, Num_CELL)  # the fast fading from the user to all the cells
      norm_distance = np.zeros(Num_CELL)
      shadow_fading = np.random.lognormal(mean=0,sigma=0.8,size=Num_CELL)
      for num in cell_id:
          # print(num)
          # print (cells[num][0] - users[0]) ** 2
          # print (cells[num][1] - users[1]) ** 2
          # print np.sqrt((cells[num][0] - users[0]) ** 2 + (cells[num][1] - users[1]) ** 2) * resolution / 20.0
          norm_distance[num] = np.sqrt((cells[num][0] - users[0]) ** 2.0 + (cells[num][1] - users[1]) ** 2) * resolution# calculate the distance between user and each base station
      large_scale = 10**((-36.7 * np.log10(norm_distance*10**-3)-149.5+shadow_fading)/10)#+10**-8
      snr =   channels_square**2 * large_scale  # assume that "p * 10^-12/noise_power = 1" is feasible
      rates = np.log2(1 + 10**7*snr)
      return rates