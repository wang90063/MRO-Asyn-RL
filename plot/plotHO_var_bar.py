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

data_list=[]
for file_ in allFiles:
    data = np.genfromtxt(file_, delimiter=',', skip_header=0,
                         skip_footer=0, names=['x', 'y'])  # ,max_rows=500
    data_list.append(data)
data2_list = []

n=0
for file2_ in allFiles2:

    data2 = np.genfromtxt(file2_, delimiter=',', skip_header=0,
                     skip_footer=0, names=['x2', 'y2'])
    data2_list.append(data2)
    n+=1
sum2 = 0
for j in range(n):
    sum2 = sum2 + sum(data2_list[j]['y2'][0:19])


data3_list = []
for file3_ in allFiles3:
     data3 = np.genfromtxt(file3_, delimiter=',', skip_header=0,
                      skip_footer=0, names=['x3', 'y3'])
     data3_list.append(data3)


ave = sum(data['y'])/1000
ave2 = sum2/n/20
print(ave2)
ave3 = sum(data3['y3'])/100
print(ave,ave3)
mean_values = [ave2,ave,ave3]
# print(data['y'],np.min(data['y']),np.max(data['y']))
# var = (np.max(data['y'])-ave)

# variance = [var]
bar_labels = ['UCB','a3c-offline','a3c-online']

x_pos = list(range(len(bar_labels)))

plt.bar(x_pos, mean_values, align='center', alpha=0.5)

plt.ylabel('Handover Rate')
plt.xticks(x_pos, bar_labels)

plt.legend()
plt.show()