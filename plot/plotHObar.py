import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from math import factorial
path ='/home/cui-group-wz/Documents/Code/MRO-Asyn-RL/plot/RL/HO'
path2 = '/home/cui-group-wz/Documents/Code/MRO-Asyn-RL/plot/BAN/HO'
path3 ='/home/cui-group-wz/Documents/Code/MRO-Asyn-RL/plot/RLonline/HO'
allFiles = glob.glob(path + "/*.csv")
allFiles2 = glob.glob(path2 + "/*.csv")
allFiles3 = glob.glob(path3 + "/*.csv")
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def savitzky_golay(y, window_size, order, deriv=0, rate=1):


    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


i=0
data_list=[]
for file_ in allFiles:
    data = np.genfromtxt(file_, delimiter=',', skip_header=0,
                         skip_footer=0, names=['x', 'y'])  # ,max_rows=500
    data_list.append(data)
    i+=1

data2_list = []
n=0
for file2_ in allFiles2:

    data2 = np.genfromtxt(file2_, delimiter=',', skip_header=0,
                     skip_footer=0, names=['x2', 'y2'])
    data2_list.append(data2)
    n+=1


data3_list = []
p=0
for file3_ in allFiles3:
     data3 = np.genfromtxt(file3_, delimiter=',', skip_header=0,
                      skip_footer=0, names=['x3', 'y3'])
     data3_list.append(data3)
     p += 1


sum= np.zeros_like(data_list[0]['y'])
sum2 = np.zeros_like(data2_list[0]['y2'])
sum3 = np.zeros_like(data3_list[0]['y3'])

for j in range(i):
    if data_list[j].shape != sum.shape:
        continue
    sum = sum+data_list[j]['y']

for j in range(n):
    print(j, data2_list[j]['y2'].shape)
    if data2_list[j].shape != sum2.shape:
        continue
    sum2 = sum2 + data2_list[j]['y2']

for j in range(p):
    sum3 = sum3 + data3_list[j]['y3']





plt.plot(data_list[0]['x'], smooth(sum/i,50), color='r', label='A3C-offine')
plt.plot(data2_list[0]['x2'], smooth(sum2/n,50), color='g', label='UCB')
plt.plot(data3_list[0]['x3']*2-5, smooth(sum3/p,50), color='b', label='A3C-online')


# plt.plot(data_list[0]['x'], savitzky_golay(sum/i, 11,0), color='r', label='A3C-offine')
# plt.plot(((data2_list[0]['x2'])), savitzky_golay(sum2/n, 51,0), color='g', label='UCB')
# plt.plot((data3_list[0]['x3']), savitzky_golay(sum3/p, 11,0), color='b', label='A3C-online')


# ave = np.sum(sum/i)/100
# ave2 = np.sum(sum2/n)/100
# ave3 = np.sum(sum3/p)/100
# mean_values = [ave,ave2,ave3]
#
# var = np.max(sum/i)-ave
# var2 = np.max(sum2/n)-ave2
# var3 = np.max(sum3/p)-ave3
# variance = [var,var2,var3]
#
# bar_labels = ['a3c-offline','UCB', 'a3c-online']
#
# x_pos = list(range(len(bar_labels)))
# plt.bar(x_pos, mean_values,yerr=variance, align='center', alpha=0.5)
#
#
# max_y = max(zip(mean_values, variance))  # returns a tuple, here: (3, 5)
# plt.ylim([0, (max_y[0] + max_y[1]) * 1.1])
#
# plt.ylabel('variable y')
# plt.xticks(x_pos, bar_labels)
# plt.title('Bar plot with error bars')
#
plt.xlabel('Episode Index')
plt.ylabel('Handover Rate')
plt.title('')
plt.legend()
plt.show()