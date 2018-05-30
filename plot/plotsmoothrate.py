import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from math import factorial
path ='/home/cui-group-wz/Documents/Code/MRO-Asyn-RL/plot/RL/rate'
# path1 = '/home/cui-group-wz/Documents/Code/MRO-Asyn-RL/plot/3gpp/rate'
path2 = '/home/cui-group-wz/Documents/Code/MRO-Asyn-RL/plot/BAN/rate'
path3 = '/home/cui-group-wz/Documents/Code/MRO-Asyn-RL/plot/RLonline/rate'
allFiles = glob.glob(path + "/*.csv")
# allFiles1 = glob.glob(path1 + "/*.csv")
allFiles2 = glob.glob(path2 + "/*.csv")
allFiles3 = glob.glob(path3 + "/*.csv")
#

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box,mode='same')
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
                     skip_footer=0, names=['x', 'y'],max_rows=7500)#
    data_list.append(data)
    i+=1

# data1_list = []
# m=0
# for file1_ in allFiles1:
#
#     data1 = np.genfromtxt(file1_, delimiter=',', skip_header=0,
#                      skip_footer=0, names=['x1', 'y1'])#
#     data1_list.append(data1)
#     m+=1

data2_list = []
n=0
for file2_ in allFiles2:

    data2 = np.genfromtxt(file2_, delimiter=',', skip_header=0,
                     skip_footer=0, names=['x2', 'y2'])
    data2_list.append(data2)
    n+=1

data3_list =[]
p=0
for file3_ in allFiles3:

    data3 = np.genfromtxt(file3_, delimiter=',', skip_header=0,
                     skip_footer=0, names=['x3', 'y3'])
    data3_list.append(data3)
    p+=1

sum= np.zeros_like(data_list[0]['y'])
# sum1 = np.zeros_like(data1_list[0]['y1'])
sum2 = np.zeros_like(data2_list[0]['y2'])
sum3 =  np.zeros_like(data3_list[0]['y3'])

for j in range(i):
    # print(j,data_list[j]['y'].shape)
    sum = sum+data_list[j]['y']
# for j in range(m):
#     sum1 = sum1 + data1_list[j]['y1']
for j in range(n):
    if data2_list[j].shape != sum2.shape:
        continue
    sum2 = sum2 + data2_list[j]['y2']
for j in range(p):
    if data3_list[j].shape != sum3.shape:
        continue
    sum3 = sum3 + data3_list[j]['y3']


print(smooth(sum3/p,50))
# plt.plot((data_list[0]['x']-5)*1.5, savitzky_golay(sum/i, 101,0), color='r', label='A3C-offline')
# # plt.plot(data1_list[0]['x1'], smooth(sum1/m,100), color='b', label='3gpp')
# plt.plot((-data2_list[0]['x2']+45)*2+105, savitzky_golay(sum2/n,95,0), color='g', label='UCB')
# plt.plot((data3_list[0]['x3']-5)*1.5, savitzky_golay(sum3/p,101,0), color='b', label='A3C-online')

plt.plot((data_list[0]['x'])*2-2, sum/i, color='r', label='A3C-offline')
# plt.plot(data1_list[0]['x1'], smooth(sum1/m,100), color='b', label='3gpp')
plt.plot((-data2_list[0]['x2'])*2+198, sum2/n, color='g', label='UCB')
x=smooth(sum3/p,50)

plt.plot((data3_list[0]['x3'])*2-2,x, color='b', label='A3C-online')

plt.xlabel('Episode Index')
plt.ylabel('Throughput (bit/s/Hz)')
plt.title('')
plt.legend()
plt.show()


