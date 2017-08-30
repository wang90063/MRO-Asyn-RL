import numpy as np

from system_model import SystemModel
from constants import ACTION_SIZE
from constants import  LOCAL_T_MAX

class bandit(object):
    def __init__(self, B):

        self.b = B
        self.k = 0
        self.model = SystemModel()
        self.model.intialize_para()
        self.mean_bs = np.random.rand(ACTION_SIZE)
        self.num_bs = np.zeros(ACTION_SIZE)
        self.t = 1
        self.csv_write_ho=[]
        self.csv_write_rate = []
        self.count_HO=0
        self.count_no_HO=0

    def _choose_action(self):
        return np.argmax(self.mean_bs + np.sqrt(2 * np.log(self.t) / ((self.t - 0.99999999) * self.num_bs)))

    def process(self):
        actions = []
        episode_rate = 0
        for i in range(1, self.b + 1):
            if i>3:
                ii=3.95
                self.k = round((2 ** (ii ** 2) - 2 ** (ii ** 2 - 1)) / ii) * ACTION_SIZE
            else:self.k = round((2 ** (i ** 2) - 2 ** (i ** 2 - 1)) / i) * ACTION_SIZE

            for j in range(0, int(self.k)):
                print('i',i,'j',j)
                action = self._choose_action()
                actions.append(action)
                episode_reward = 0

                self.model.state_update(actions[j-1], actions[j])
                reward = self.model.reward
                episode_reward += reward

                episode_rate += self.model.rate

                self.model.update()

                if actions[j-1]!= actions[j]:
                    self.count_HO +=1.0
                else:self.count_no_HO+=1.0


                for m in range(1,i):
                    self.model.state_update(actions[j], actions[j])
                    reward = self.model.reward
                    episode_reward += reward
                    episode_rate += self.model.rate
                    self.model.update()


                self.num_bs[actions[j]] += i
                self.mean_bs[actions[j]] = self.mean_bs[actions[j]] + 1 / self.num_bs[actions[j]] * (
                    episode_reward - i * self.mean_bs[actions[j]])
                self.t += i
                # self.handover_ratio = self.count_HO / (
                #     self.count_HO + self.count_no_HO + 1.0)
                self.handover_ratio = self.model.count_handover_total / (
                    self.model.count_no_handover + self.model.count_handover_total + 1)
                self.csv_write_ho.append([self.t, self.handover_ratio])

                self.csv_write_rate.append([self.t, episode_rate /self.t])
                self.model.init_users()


            self.count_HO = 0
            self.count_no_HO = 0
            self.model.init_users()

