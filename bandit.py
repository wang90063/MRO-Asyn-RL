import numpy as np

from system_model import SystemModel
from constants import ACTION_SIZE


class bandit(object):
    def __init__(self, B):


        self.b = B
        self.k = 0
        self.model = SystemModel()
        self.model.intialize_para()
        self.mean_bs = np.random.rand(ACTION_SIZE)
        self.num_bs = np.ones(ACTION_SIZE)
        self.t =1
        self.csv_write_ho=[]
        self.csv_write_rate = []
        self.count_HO=0
        self.count_no_HO=0



    def _choose_action(self):
        return np.argmax(self.mean_bs + np.sqrt(2 * np.log(self.t) / ((self.num_bs))))

    def set_start_time(self, start_time):
        self.start_time = start_time


    def process(self):

        # episode_rate = 0
        #
        # action = 0
        #
        # count_ho=0
        #
        # for i in range(1,self.b+1):
        #     self.k = round((2 ** (i ** 2) - 2 ** (i ** 2 - 1)) / i) * ACTION_SIZE
        #
        #     for j in range(0,int(self.k)):
        #
        #         last_action = action
        #         action = self._choose_action()
        #
        #         if action != last_action:
        #
        #             count_ho +=1.0
        #
        #             print("ho time step", self.t)
        #
        #         stay_rate = 0
        #         for m in range(1,i):
        #
        #             self.model.state_update(action,action)
        #
        #             stay_rate+=self.model.rate
        #
        #             episode_rate +=self.model.rate
        #
        #             self.model.update()
        #
        #         self.num_bs[action] += i
        #         self.mean_bs[action] = self.mean_bs[action] + 1 / self.num_bs[action] * (
        #                 stay_rate - i * self.mean_bs[action])
        #         self.t += i
        #
        #
        #
        #         if self.t > 4000000:
        #
        #             print('time step', self.t, "ho rate", count_ho/self.t, "rate", episode_rate/self.t)
        #             break
        #
        #     if self.t >4000000:
        #
        #        break

        actions = []
        episode_rate = 0
        self.episode_counter=1

        for j in range(0, 140000):

                action = self._choose_action()

                actions.append(action)
                episode_reward = 0



                self.model.state_update(actions[j-1], actions[j])


                if self.model.terminal == True:
                    self.model.init_users()
                    count = 0


                reward = self.model.reward
                episode_reward += reward

                episode_rate += self.model.rate

                self.model.update()


                if actions[j-1]!= actions[j]:
                    # print(self.t)
                    self.count_HO +=1.0
                else:self.count_no_HO+=1.0


                if self.model.terminal == True:
                    self.model.init_users()
                    count = 0

                self.num_bs[actions[j]] += 1
                self.mean_bs[actions[j]] = self.mean_bs[actions[j]] + 1 / self.num_bs[actions[j]] * (
                    episode_reward - 1 * self.mean_bs[actions[j]])
                self.t += 1


                if self.t>=140000:
                    print(self.t, self.model.count_handover_total,
                      self.model.count_handover_total / self.t, episode_rate / self.t)
                    self.csv_write_ho.append([self.episode_counter,self.model.count_handover_total / self.t])
                    self.csv_write_rate.append([self.episode_counter, episode_rate / self.t])
                    break



