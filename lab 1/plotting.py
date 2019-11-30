import matplotlib.pyplot as plt
import csv

x = []
y = []

filename='value_sarsa8.txt'
# with open(filename,'r') as csvfile:
#     plots = csv.reader(csvfile, delimiter=',')
#     for row in plots:
#         x.append(int(row[0]))
#         y.append(int(row[1]))

import numpy as np

time,e,w,l,r = np.loadtxt(filename, delimiter=',', unpack=True)
plt.plot(time,r, label='Value function graph')

plt.xlabel('time')
plt.ylabel('value function')
plt.title('Value function vs time')
plt.legend()
plt.show()
