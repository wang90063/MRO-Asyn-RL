import matplotlib.pyplot as plt
import numpy as np
import glob

path ='/home/cui-group-wz/Documents/Code/MRO-Asyn-RL/plot/RL/rate'
path1 = '/home/cui-group-wz/Documents/Code/MRO-Asyn-RL/plot/3gpp/rate'
path2 = '/home/cui-group-wz/Documents/Code/MRO-Asyn-RL/plot/BAN/rate'
path3 = '/home/cui-group-wz/Documents/Code/MRO-Asyn-RL/plot/RLonline/rate'
allFiles = glob.glob(path + "/*.csv")
allFiles1 = glob.glob(path1 + "/*.csv")
allFiles2 = glob.glob(path2 + "/*.csv")
allFiles3 = glob.glob(path3 + "/*.csv")

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

i=0
data_list=[]
for file_ in allFiles:

    data = np.genfromtxt(file_, delimiter=',', skip_header=0,
                     skip_footer=0, names=['x', 'y'],max_rows=7500)#
    data_list.append(data)
    i+=1

data1_list = []
m=0
for file1_ in allFiles1:

    data1 = np.genfromtxt(file1_, delimiter=',', skip_header=0,
                     skip_footer=0, names=['x1', 'y1'])#
    data1_list.append(data1)
    m+=1

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
sum1 = np.zeros_like(data1_list[0]['y1'])
sum2 = np.zeros_like(data2_list[0]['y2'])
sum3 =  np.zeros_like(data3_list[0]['y3'])

for j in range(i):
    print(j,data_list[j]['y'].shape)
    sum = sum+data_list[j]['y']
for j in range(m):
    sum1 = sum1 + data1_list[j]['y1']
for j in range(n):
    sum2 = sum2 + data2_list[j]['y2']
for j in range(p):
    sum3 = sum3 + data3_list[j]['y3']


plt.plot(data_list[0]['x']+137000, smooth(sum/i,220), color='r', label='A3C-offline')
# plt.plot(data1_list[0]['x1'], smooth(sum1/m,100), color='b', label='3gpp')
plt.plot(data2_list[0]['x2'], smooth(sum2/n,1), color='g', label='UCB')
plt.plot(data3_list[0]['x3']-13000, smooth(sum3/p,220), color='b', label='A3C-online')

plt.xlabel('global time step')
plt.ylabel('throughput')
plt.title('')
plt.legend()
plt.show()


