import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 

events_qpsk = np.loadtxt('dataset/events_qpsk.csv')
events_16qam = np.loadtxt('dataset/events_16qam.csv')
events_64qam = np.loadtxt('dataset/events_64qam.csv')
events_256qam = np.loadtxt('dataset/events_256qam.csv')

snr = np.arange(-39.9,40.1,0.1)

plt.plot(snr, events_qpsk, linestyle='solid', color='black')
plt.plot(snr, events_16qam, linestyle='solid', color='red')
plt.plot(snr, events_64qam, linestyle='solid', color='green')
plt.plot(snr, events_256qam, linestyle='solid', color='blue')
plt.legend(['QPSK', '16-QAM', '64-QAM', '256-QAM'])
plt.title('External Events')
plt.xlim((40, -40))
plt.grid(True)
plt.show()

measured_qpsk = np.loadtxt('dataset/measured_qpsk.csv')
# Ploting values
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(measured_qpsk, events_qpsk, snr, zdir='z', label='Relation between Events and Measured SNR')
ax.set_xlabel('measured', fontsize=14)
ax.set_ylabel('events', fontsize=14)
ax.set_zlabel('$y$', fontsize=14)
plt.show()