import numpy as np
from matplotlib import pyplot

# Preraring dataset
y1 = np.loadtxt('dataset/measured_qpsk.csv')
y2 = np.loadtxt('dataset/measured_16qam.csv')
y3 = np.loadtxt('dataset/measured_64qam.csv')
y4 = np.loadtxt('dataset/measured_256qam.csv')

n = len(y1)

x = np.arange(-39.9,40.1,0.1)

x1 = np.zeros(n)
x2 = np.zeros(n)
x3 = np.zeros(n)
x4 = np.zeros(n)

x1 = x - y1
x2 = x - y2
x3 = x - y3
x4 = x - y4

x1 = x1 + 0.035*np.random.randn(n) #0.025*np.random.randn(n)
x2 = x2 + 0.035*np.random.randn(n) #0.025*np.random.randn(n)
x3 = x3 + 0.035*np.random.randn(n) #0.025*np.random.randn(n)
x4 = x4 + 0.035*np.random.randn(n) #0.025*np.random.randn(n)

np.savetxt('dataset/diff_qpsk.csv', x1)
np.savetxt('dataset/diff_16qam.csv', x2)
np.savetxt('dataset/diff_64qam.csv', x3)
np.savetxt('dataset/diff_256qam.csv', x4)

pyplot.plot(y1[300:], x1[300:], '.', color="blue")
pyplot.plot(y2[300:], x2[300:], '.', color="green")
pyplot.plot(y3[300:], x3[300:],'.', color="black")
pyplot.plot(y4[300:], x4[300:], '.', color="red")
pyplot.xticks(np.arange(-20,45,5))
pyplot.legend(['QPSK', '16QAM', '64QAM', '256QAM'])
pyplot.show()
