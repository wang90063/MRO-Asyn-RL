import tensorflow as tf
import numpy as np
from system_model import SystemModel
from constants import slot
from constants import ACTION_SIZE
from constants import num_test_data
model = SystemModel()
from constants import num_train_data
from constants import batch_size
batch_num = num_train_data/batch_size
def choose_action(last_action,threshold):
    if np.max(model.rates) - model.rates[last_action] > threshold:
       return np.argmax(model.rates)
    else:
       return last_action
# for m in range(PARALLEL_SIZE):
model.intialize_para()
   # m=0
for m in range(slot):
  threshold = 4.5+m/slot
  actions = []
  states = []
  i=0
  while 1!=0:
      if i==0:
         action = choose_action(model.last_serve_cell_id,threshold)
         actions.append(action)
      else:
         action = choose_action(actions[i-1],threshold)
         actions.append(action)

      for j in range(2 * ACTION_SIZE):
           states.append(model.s_t[j])

      model.state_update(actions[i - 1], actions[i])
      model.update()

      if model.terminal ==True:
          break
  # for i in range(num_train_data/slot):
  #   if i == 0:
  #     action = choose_action(model.last_serve_cell_id,threshold)
  #     actions.append(action)
  #   else:
  #     action = choose_action(actions[i-1],threshold)
  #     actions.append(action)
  #   for j in range(2 * ACTION_SIZE):
  #       states.append(model.s_t[j])
  #   model.state_update(actions[i - 1], actions[i])
  #   model.update()

  with open("predata/data.txt",'a') as f:
     for state in states:
          f.write(str(state)+'\n')
     f.close()

  with open("predata/label.txt",'a') as g:
      for action in actions:
         g.write(str(action)+'\n')
      g.close()


model_test = SystemModel()
model_test.intialize_para()
for m in range(slot):
  threshold = 4.5+m/slot
  model.init_users()
  states_test = []
  actions_test = []



  for i in range(num_test_data/slot):
    if i == 0:
      action_test_c = choose_action(model_test.last_serve_cell_id,threshold)
      actions_test.append(action_test_c)
    else:
      action_test_c = choose_action(actions_test[i-1],threshold)
      actions_test.append(action_test_c)
    for j in range(2 * ACTION_SIZE):
        states_test.append(model.s_t[j])
    model_test.state_update(actions_test[i - 1], actions_test[i])
    model_test.update()

  with open("predata/data_test.txt",'a') as k:
     for state_test in states_test:
          k.write(str(state_test)+'\n')
     k.close()

  with open("predata/label_test.txt",'a') as p:
      for action_test in actions_test:
         p.write(str(action_test)+'\n')
      p.close()