from bandit import bandit
from constants import PLOT_BAN_HO_DIR
from constants import PLOT_BAN_rate_DIR
from constants import B
from constants import num_ite
import numpy as np

for ite in range( num_ite):
    ban = bandit(B)
    ban.process()

    write_plot_ho = np.asarray(ban.csv_write_ho)
    fname = PLOT_BAN_HO_DIR + '/' + 'HO' + 'ite' + '-' + str(ite+800) + '.csv'
    np.savetxt(fname, write_plot_ho, delimiter=",")

    write_plot_rate = np.asarray(ban.csv_write_rate)
    gname = PLOT_BAN_rate_DIR + '/' + 'HO' + 'ite' + '-' + str(ite+800) + '.csv'
    np.savetxt(gname, write_plot_rate, delimiter=",")


