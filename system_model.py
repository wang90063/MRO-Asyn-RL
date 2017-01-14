import pygame
import sys
import random
import numpy as np
from pygame.locals import *
from constants import LOCAL_T_MAX
from collections import Counter
from math import exp

# frame params
FPS = 30  # frames per second, the general speed of the program

# 400 pixels represent the largest distance of the area, i.e. 100m
WINDOW_WIDTH = 400  # size of window's width in pixels
WINDOW_HEIGHT = 400  # size of windows' height in pixels
cell_radius = 25 # "m" this is the radius of cell
resolution  = cell_radius * 4./WINDOW_WIDTH # meter/pixel, the longest distace in the simulation system is "cell_radius * 4"

outlayer_userrange_x_low = 24.5 / resolution
outlayer_userrange_x_high = 75.5 / resolution

outlayer_userrange_y_low = 20 / resolution
outlayer_userrange_y_high = 80 / resolution

innerlayer_userange_x_low = 33 / resolution
innerlayer_userange_x_high = 67 / resolution

innerlayer_userange_y_low = 30 / resolution
innerlayer_userange_y_high = 70 / resolution

Num_CELL = 7
NUM_USER = 1 # In asynchronous deep Q learning, only one user in one thread

# RGB
GRAY = (100, 100, 100)
NAVY_BLUE = (60, 60, 100)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 128, 0)
PURPLE = (255, 0, 255)
CYAN = (0, 255, 255)
background_color = WHITE


# generate cell list, every cell in the lists consists of 5 params:
# (1)loc_x(left) (2)loc_y(top) (3)cell_id
# :return: cell_list: (1)loc_x(left) (2)loc_y(top) (3)cell_id

# cell location
cell_id = np.arange(Num_CELL)

# the locations of cells are fixed and the coordinates are given
cell_x = [200, 200, 370, 370, 200, 30, 30]
cell_y = [200, 0, 100, 300, 400, 300, 100]

cells = np.vstack((cell_x, cell_y))


pygame.init()
FPSCLOCK = pygame.time.Clock()
DISPLAY_SURF = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('MRO System Model')



# "action" is the "cell_id" selected by the agent
class SystemModel:
  def __init__(self):
      self.users, self.serve_cell_id = init_users()
      self.handover_indicator = np.zeros(LOCAL_T_MAX)

  def frame_step(self, users, action):

      reward = self._get_reward(users, action)
      rates = get_rate_percell(users, cells)
      max_num = np.argmax(rates)
      feature_user_rates_vector = np.zeros(1, Num_CELL)
      feature_user_rates_vector[max_num] = 1
      feature_handover = np.zeros(1,2)
      if self.reward_handover ==0 or self.reward_handover ==1:
          feature_handover[0] = 1
      else:
          feature_handover[1] = 1
      s_t = np.hstack((feature_user_rates_vector,feature_handover))
      return reward, s_t
  def _move_user(self, users):
      """
      low mobility users are considered, i.e. the user only move one pixel every frame. different mobility trajectories will be tested to present the robustness of the neural network
      """
      mobility_speed = 1
      move_x = random.randint(-mobility_speed, mobility_speed)
      user_x_tmp = users[:, 0] + move_x
      move_y = random.randint(-mobility_speed, mobility_speed)
      user_y_tmp = users[:, 1] + move_y

      if user_x_tmp > innerlayer_userange_x_low and user_y_tmp < innerlayer_userange_x_high and user_y_tmp > innerlayer_userange_y_low and user_y_tmp < innerlayer_userange_y_high:
          self.user_x = users[:, 0] - move_x
          self.user_y = users[:, 1] - move_y
      else:
          self.user_x = user_x_tmp
          self.user_y = user_y_tmp

      return users

  def _get_reward(self, users, action):
      """
      :param users: the location of user before moving
      :param action: the taken action to obtain "users"
      :return: reward : the weighted sum of rate and reward for handover, i.e. "handover error occurs" -- 0, "handover successes" -- 1
      """
      reward_weight_rate = 0.5
      self.users_move = self._move_user(users)
      self.rates_move = get_rate_percell(self.users_move, cells)
      rate = self.rates_move[action]

      self.count_handover = 0

      if action == self.serve_cell_id:
          self.reward_handover = 0
          self.count_handover += 1
          self.handover_indicator[self.count_handover] = 0
      elif self.handover_indicator[0] == 1:
          self.reward_handover = -1
          self.count_handover = 0
          self.handover_indicator[self.count_handover] = 1
      else:
          self.reward_handover = 1
      self.serve_cell_id = np.copy(action)

      reward = reward_weight_rate * rate + (1-reward_weight_rate) * self.reward_handover
      return reward


def init_users():
    """
    initialize user. every user consists of 4 params:
    (1) loc_x(center) (2) loc_y(center) (3) which cell user is in (4) user mobility type
    user mobility type is divided into 3 categories: low, medium and high. Low mobility users takes 70% of all,
    while medium and high takes 20% and 10%.
    :return: user: (1) loc_x(center) (2) loc_y(center) (3) which cell user is in (4) user mobility type
    """
    while True:
        user_x_tmp = np.random.randint(outlayer_userrange_x_low, outlayer_userrange_x_high+1, size=NUM_USER, dtype='int')
        user_y_tmp = np.random.randint(outlayer_userrange_y_low, outlayer_userrange_y_high+1, size=NUM_USER, dtype='int')
        if user_x_tmp < innerlayer_userange_x_low or user_x_tmp > innerlayer_userange_x_high or user_y_tmp < innerlayer_userange_y_low or user_y_tmp > innerlayer_userange_y_high:
           user_x = user_x_tmp
           user_y = user_y_tmp
           break
    users = np.vstack((user_x, user_y))
    rates = get_rate_percell(users, cells)
    serve_cell_id= np.argmax(rates)
    return users.T, serve_cell_id

def get_rate_percell(users, cells):
      """
      get the rates of the user in all the cells if this user connects to the cell. return the array "rate" to represent the rate in the cells
      """
      channels_square = np.random.rayleigh(1, (1, Num_CELL)) ** 2  # the fast fading from the user to all the cells
      norm_distance = np.array(
          np.sqrt((cells[num][0] - users[0]) ** 2 + (cells[num][1] - users[1]) ** 2) * resolution / 20.0 for num in
          cell_id)  # calculate the distance between user and each base station
      snr = channels_square * (norm_distance ** -4)  # assume that "p * 10^-12/noise_power = 1" is feasible
      rates = np.log(1 + snr)
      return rates