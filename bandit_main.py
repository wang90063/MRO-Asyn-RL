from bandit import bandit
from constants import PLOT_BAN_HO_DIR
from constants import PLOT_BAN_rate_DIR
from constants import B
from constants import num_ite
import numpy as np
import time


for ite in range(500):
    print(ite)
    start_time = time.time()
    ban = bandit(B)
    ban.process()

    elapsed_time = time.time() - start_time

    print(elapsed_time)
    write_plot_ho = np.asarray(ban.csv_write_ho)
    fname = PLOT_BAN_HO_DIR + '/' + 'HO' + 'ite' + '-' + str(ite) + '.csv'
    np.savetxt(fname, write_plot_ho, delimiter=",")

    write_plot_rate = np.asarray(ban.csv_write_rate)
    gname = PLOT_BAN_rate_DIR + '/' + 'rate' + 'ite' + '-' + str(ite) + '.csv'
    np.savetxt(gname, write_plot_rate, delimiter=",")