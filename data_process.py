import numpy as np
import matplotlib.pyplot as plt

file_name = 'data_2'
f_2 = 'data'
a = np.loadtxt('data_1124_2m',dtype = float)
b = np.loadtxt(f_2,dtype = float)
print(len(a))
#a = a[:1000000]
lst = [x for x in range(1000000)]
#print(len(a))

plt.figure(figsize = (7,5))

plt.grid(True)
plt.xlabel('Timesteps')
plt.ylabel('Accumulated reward')
#plt.legend()
#plt.xlim(0.5,1030000.5)
#plt.ylim(-1000,350000)
plt.plot(lst,a,label = 'BCQ',linestyle = '--',lw = 3,alpha = 0.7)
#plt.plot(lst,b,label = 'DDPG', lw = 3,alpha = 0.7)
plt.legend()
plt.title('Result')
plt.show()