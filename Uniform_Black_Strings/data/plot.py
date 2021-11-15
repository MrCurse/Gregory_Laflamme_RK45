import numpy as np 
import matplotlib.pyplot as plt 

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)

data1 = np.loadtxt('GL_D_equals_4.txt')
data2 = np.loadtxt('GL_D_equals_5.txt')
data3 = np.loadtxt('GL_D_equals_6.txt')
data4 = np.loadtxt('GL_D_equals_7.txt')
data5 = np.loadtxt('GL_D_equals_8.txt')
data6 = np.loadtxt('GL_D_equals_9.txt')

plt.scatter(data1[:,0], data1[:,1], edgecolor='black')
plt.scatter(data2[:,0], data2[:,1], edgecolor='black')
plt.scatter(data3[:,0], data3[:,1], edgecolor='black')
plt.scatter(data4[:,0], data4[:,1], edgecolor='black')
plt.scatter(data5[:,0], data5[:,1], edgecolor='black')
plt.scatter(data6[:,0], data6[:,1], edgecolor='black')
plt.xlabel(r'$\mu$', fontsize=16)
plt.ylabel(r'$\Omega$', fontsize=16)
plt.savefig('dispersion.png', dpi=300, bbox_inches='tight')
plt.show()