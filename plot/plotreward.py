import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from io import StringIO




path = '/home/cui-group-wz/Documents/Code/MRO-Journal/plot/reward/5/HO'
path2 ='/home/cui-group-wz/Documents/Code/MRO-Journal/plot/reward/10/HO'
path3 ='/home/cui-group-wz/Documents/Code/MRO-Journal/plot/reward/15/HO'#-noclustering
path4 ='/home/cui-group-wz/Documents/Code/MRO-Journal/plot/reward/20/HO'
path5 ='/home/cui-group-wz/Documents/Code/MRO-Journal/plot/reward/25/HO'#-noclustering
path6 ='/home/cui-group-wz/Documents/Code/MRO-Journal/plot/reward/35/HO'#-noclustering
allFiles = glob.glob(path + "/*.csv")
allFiles2 = glob.glob(path2 + "/*.csv")
allFiles3 = glob.glob(path3 + "/*.csv")
allFiles4 = glob.glob(path4 + "/*.csv")
allFiles5 = glob.glob(path5 + "/*.csv")
allFiles6 = glob.glob(path6 + "/*.csv")

path_rate = '/home/cui-group-wz/Documents/Code/MRO-Journal/plot/reward/5/rate'
path2_rate = '/home/cui-group-wz/Documents/Code/MRO-Journal/plot/reward/10/rate'
path3_rate ='/home/cui-group-wz/Documents/Code/MRO-Journal/plot/reward/15/rate'
path4_rate = '/home/cui-group-wz/Documents/Code/MRO-Journal/plot/reward/20/rate'
path5_rate ='/home/cui-group-wz/Documents/Code/MRO-Journal/plot/reward/25/rate'
path6_rate ='/home/cui-group-wz/Documents/Code/MRO-Journal/plot/reward/35/rate'
allFiles_rate = glob.glob(path_rate + "/*.csv")
allFiles2_rate = glob.glob(path2_rate + "/*.csv")
allFiles3_rate = glob.glob(path3_rate + "/*.csv")
allFiles4_rate = glob.glob(path4_rate + "/*.csv")
allFiles5_rate = glob.glob(path5_rate + "/*.csv")
allFiles6_rate = glob.glob(path6_rate + "/*.csv")
# HO
############################
i = 0

data_list=[]
for file_ in allFiles:
    data = np.genfromtxt(file_, delimiter=',', skip_header=0,
                         skip_footer=0, names=['x', 'y'])  # ,max_rows=500
    data_list.append(data)
    i+=1

sum_0 = 0

for j in range(i):
    sum_0 = sum_0 + data_list[j]['y']

ave = sum_0/i

##############################


############################################

data2_list = []

n=0
for file2_ in allFiles2:

    data2 = np.genfromtxt(file2_, delimiter=',', skip_header=0,
                     skip_footer=0, names=['x2', 'y2'])
    data2_list.append(data2)
    n+=1
sum2 = 0
for j in range(n):
    sum2 = sum2 + data2_list[j]['y2']#sum([0:69])
ave2 = sum2/n
################################################

##############################################

data3_list = []
m=0
for file3_ in allFiles3:
     data3 = np.genfromtxt(file3_, delimiter=',', skip_header=0,
                      skip_footer=0, names=['x3', 'y3'])
     data3_list.append(data3)
     m+=1
sum3 = 0
for j in range(m):
    sum3 = sum3 + data3_list[j]['y3']
ave3 = sum3/m


########################################################

########################################################
data4_list = []
p=0
for file4_ in allFiles4:
     data4 = np.genfromtxt(file4_, delimiter=',', skip_header=0,
                      skip_footer=0, names=['x4', 'y4'])
     data4_list.append(data4)
     p+=1
sum4 = 0
for j in range(p):
    sum4 = sum4 + data4_list[j]['y4']
ave4 = sum4/p

##########################################################

data5_list = []
q=0
for file5_ in allFiles5:
     data5 = np.genfromtxt(file5_, delimiter=',', skip_header=0,
                      skip_footer=0, names=['x5', 'y5'])
     data5_list.append(data5)
     q+=1
sum5 = 0
for j in range(q):
    sum5 = sum5 + data5_list[j]['y5']
ave5 = sum5/q

#######################################################

##########################################################

data6_list = []
q=0
for file6_ in allFiles6:
     data6 = np.genfromtxt(file6_, delimiter=',', skip_header=0,
                      skip_footer=0, names=['x6', 'y6'])
     data6_list.append(data6)
     q+=1
sum6 = 0
for j in range(q):
    sum6 = sum6 + data6_list[j]['y6']
ave6 = sum6/q

#######################################################

####################################################

#rate
##########################################################
data_list_rate=[]
i = 0
for file_ in allFiles_rate:

    data_rate = np.genfromtxt(file_, delimiter=',', skip_header=0,
                     skip_footer=0, names=['x', 'y'],max_rows=7500)#
    data_list_rate.append(data_rate)
    i+=1
sum_rate = 0

for j in range(i):
    sum_rate = sum_rate + data_list_rate[j]['y']
ave_rate = sum_rate / i
###################################################################


#####################################################################
data2_list_rate = []
n=0
for file2_ in allFiles2_rate:

    data2_rate = np.genfromtxt(file2_, delimiter=',', skip_header=0,
                     skip_footer=0, names=['x2', 'y2'])
    data2_list_rate.append(data2_rate)
    n+=1
sum2_rate = 0
for j in range(n):
    sum2_rate = sum2_rate +data2_list_rate[j]['y2']#sum([0:69])
ave2_rate = sum2_rate/n
#############################################################################

################################################################################
data3_list_rate = []
m=0
for file3_ in allFiles3_rate:
     data3_rate = np.genfromtxt(file3_, delimiter=',', skip_header=0,
                      skip_footer=0, names=['x3', 'y3'])
     data3_list_rate.append(data3_rate)
     m+=1
sum3_rate = 0
for j in range(m):
    sum3_rate = sum3_rate + data3_list_rate[j]['y3']
ave3_rate = sum3_rate / m
#################################################################################

#################################################################################


data4_list_rate = []
p=0
for file4_ in allFiles4_rate:
     data4_rate = np.genfromtxt(file4_, delimiter=',', skip_header=0,
                      skip_footer=0, names=['x4', 'y4'])
     data4_list_rate.append(data4_rate)
     p+=1
sum4_rate = 0
for j in range(p):
    sum4_rate = sum4_rate + data4_list_rate[j]['y4']
ave4_rate = sum4_rate / p

#############################################################################

##############################################################################

data5_list_rate = []
q=0
for file5_ in allFiles5_rate:
     data5_rate = np.genfromtxt(file5_, delimiter=',', skip_header=0,
                      skip_footer=0, names=['x5', 'y5'])
     data5_list_rate.append(data5_rate)
     q+=1
sum5_rate = 0
for j in range(q):

    sum5_rate = sum5_rate + data5_list_rate[j]['y5']
ave5_rate = sum5_rate / q

################################################################################

##############################################################################

data6_list_rate = []
q=0
for file6_ in allFiles6_rate:
     data6_rate = np.genfromtxt(file6_, delimiter=',', skip_header=0,
                      skip_footer=0, names=['x6', 'y6'])
     data6_list_rate.append(data6_rate)
     q+=1
sum6_rate = 0
for j in range(q):
    sum6_rate = sum6_rate + data6_list_rate[j]['y6']
ave6_rate = sum6_rate / q

################################################################################
mean_values = [ave,ave2,ave3,ave4,ave5,ave6]#
mean_values_rate = [ave_rate,ave2_rate,ave3_rate,ave4_rate,ave5_rate,ave6_rate]#*1.7*1.7

print(mean_values)
print(mean_values_rate)



raw_data = {'first_name': [r'$\beta$=5', r'$\beta$=10',r'$\beta$=15', r'$\beta$=20',r'$\beta$=25'],
        'HO_rate': [0.00042329124543761322, 0.00027902570712686224,0.00027403775380653289,0.00027066900408121513,0.0002638516237847746],
        'Throughput':[0.68522930137688776, 0.68438578127874827, 0.68389233012982731, 0.68337924456844057, 0.68224726039627941,]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'HO_rate', 'Throughput'])

pos = list(range(len(df['HO_rate'])))
width = 0.25

# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))
ax2 = ax.twinx()
# Create a bar with pre_score data,
# in position pos,
ax.bar(pos,
        #using df['pre_score'] data,
        df['HO_rate'],
        # of width
        width,
        # with alpha 0.5
        alpha=1,
        # with color
        color='blue',
        # with label the first value in first_name
        label='HO rate')


ax2.bar([p + width for p in pos],
        #using df['mid_score'] data,
        df['Throughput'],
        # of width
        width,
        # with alpha 0.5
        alpha=1,
        # with color
        color='red',
        # with label the second value in first_name
        label='Throughput')

# # Set the position of the x ticks
ax.set_xticks([p + 0.5 * width for p in pos])
#
# # Set the labels for the x ticks
ax.set_xticklabels(df['first_name'])
#
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines+lines2 , labels + labels2, loc=1)#


# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*6)
ax2.set_ylim([0.62,0.7])
ax.set_ylabel('Handover Rate')
ax2.set_ylabel('Throughput (bit/s/Hz)')


plt.show()



#
# plt.plot([0.68522930137688776, 0.68488578127874827, 0.68407924456844057,0.68339233012982731,   0.6824726039627941, 0.6813],[0.00042329124543761322, 0.00033245170778376171, 0.00027866900408121513,0.0002703775380653289,  0.0002638516237847746, 0.00025834047299597509], color='b',marker='o',markerfacecolor ='none',label='1')
# # plt.annotate (r'$\beta$=5', xy=(0.685,0.000423),xytext=(0.684,0.000423),arrowprops=dict(facecolor='black',width=1,headwidth=5,shrink=0.08))
# # plt.annotate (r'$\beta$=15', xy=(0.684,0.000279),xytext=(0.683,0.00029),arrowprops=dict(facecolor='black',width=1,headwidth=5,shrink=0.08))
# # plt.annotate (r'$\beta$=25', xy=(0.682,0.000264),xytext=(0.6815,0.00029),arrowprops=dict(facecolor='black',width=1,headwidth=5,shrink=0.08))
# # plt.annotate (r'$\beta$=35', xy=(0.68,0.000258),xytext=(0.6805,0.00029),arrowprops=dict(facecolor='black',width=1,headwidth=5,shrink=0.08))
# plt.xlabel(' Rate (bit/s/Hz)')
# plt.ylabel('Handover Rate')
# plt.title('')
#
# plt.show()

