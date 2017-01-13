import pygame
import sys
import random
import numpy as np
from pygame.locals import *
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

pygame.init()
FPSCLOCK = pygame.time.Clock()
DISPLAY_SURF = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('MRO System Model')

# "action" is the "cell_id" selected by the agent
class SystemModel:
    def __init__(self, action):
        self.users = init_users(action)
        self.cells = init_cells(self)

    def frame_step(self, input_action):
        pygame.event.pump()
        # update PRB according to actions
        # for i in range(NUM_CELL):
        #     # take the PRB from cell j to i
        #     cell_to_take = input_action[i]
        #     self.cells[i][3] += 1
        #     self.cells[cell_to_take][3] -= 1
        if input_action[1] == 1:
            if self.cells[0][3] < 20:
                self.cells[0][3] += 1
            if self.cells[1][3] > 1:
                self.cells[1][3] -= 1
        elif input_action[2] == 1:
            if self.cells[1][3] < 20:
                self.cells[1][3] += 1
            if self.cells[0][3] > 1:
                self.cells[0][3] -= 1
        print(self.cells[:, 3])
        # get reward
        reward = get_reward(self.cells)

        # update system states
        self.users = move_user(self.users)

        # draw frame
        DISPLAY_SURF.fill(background_color)
        outrage_ratio = [x[4] / x[3] for x in self.cells]
        outrage_ratio = [min(x, 1) for x in outrage_ratio]  # larger than 1 is outrage, use black color directly
        color_index = [int(x * len(COLOR_LIST)) for x in outrage_ratio]
        for cell in self.cells:
            this_color_index = color_index[cell[2]]
            try:
                this_cell_color = [i * 255 for i in list(COLOR_LIST[this_color_index - 1].rgb)]
            except:
                print("this_color_index: ", this_color_index)
                print("COLOR_LIST length: ", len(COLOR_LIST))
            pygame.draw.rect(DISPLAY_SURF, this_cell_color, (cell[0], cell[1], CELL_SIZE, CELL_SIZE))

        # draw users
        for user in self.users:
            pygame.draw.circle(DISPLAY_SURF, RED, (user[0], user[1]), 2)

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        return image_data, reward


def init_users(self, action):
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
           self.user_x = user_x_tmp
           self.user_y = user_y_tmp
           break
    servingcell_id = action
    users = np.vstack((self.user_x, self.user_y, servingcell_id))
    return users.T


def init_cells(self):
    """
    initialize cell list, every cell in the lists consists of 5 params:
    (1)loc_x(left) (2)loc_y(top) (3)cell_id
    :return: cell_list: (1)loc_x(left) (2)loc_y(top) (3)cell_id    """
    # cell location
    self.cell_id = np.arange(Num_CELL)

    # the locations of cells are fixed and the coordinates are given
    cell_x = [200, 200, 370, 370, 200, 30, 30]
    cell_y = [200, 0, 100, 300, 400, 300, 100]

    cells = np.vstack((cell_x, cell_y, self.cell_id))
    return cells.T.astype(int)


def _get_rate_percell(self, users, cells):
    """
    get the rates of the user in all the cells if this user connects to the cell. return the array "rate" to represent the rate in the cells
    """
    channels_square = np.random.rayleigh(1,(1,Num_CELL))**2 # the fast fading from the user to all the cells
    norm_distance = [np.sqrt((cells[num][0] - users[0])**2 + (cells[num][1] - users[1])**2)*resolution/20.0 for num in self.cell_id] # calculate the distance between user and each base station
    snr = channels_square * (norm_distance ** -4) # assume that "p * 10^-12/noise_power = 1" is feasible
    rates = np.log(1+snr)

    return rates


def move_user(self,users,action):
    """
    low mobility users are considered, i.e. the user only move one pixel every frame. different mobility trajectories will be tested to present the robustness of the neural network
    """
    mobility_speed = 1
    move_x = random.randint(-mobility_speed, mobility_speed)
    user_x_tmp = users[:, 0] + move_x
    move_y = random.randint(-mobility_speed, mobility_speed)
    user_y_tmp = users[:, 1] + move_y

    if user_x_tmp > innerlayer_userange_x_low and user_y_tmp < innerlayer_userange_x_high and user_y_tmp > innerlayer_userange_y_low and user_y_tmp < innerlayer_userange_y_high:
        self.user_x = users[:,0] - move_x
        self.user_y = users[:,1] - move_y
    else:
        self.user_x = user_x_tmp
        self.user_y = user_y_tmp
    users[:, 2] = action
    return users


def get_reward(cells):


    return reward


def main():
    system_model = SystemModel()
    while True:
        items = range(NUM_CELL)
        random_action = random.sample(items, len(items))
        imagedata, reward = system_model.frame_step(random_action)
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYUP and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()


if __name__ == '__main__':
    main()
