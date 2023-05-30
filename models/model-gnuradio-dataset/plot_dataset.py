import numpy as np
import pandas
from matplotlib import pyplot

# Preraring dataset
#y1 = np.loadtxt('dataset/measured_qpsk.csv')
#y2 = np.loadtxt('dataset/measured_16qam.csv')
#y3 = np.loadtxt('dataset/measured_64qam.csv')
y4 = np.loadtxt('dataset/measured_256qam.csv')

n = len(y4)

x = np.arange(-39.9,40.1,0.1)

x1 = np.zeros(n)
x2 = np.zeros(n)
x3 = np.zeros(n)
x4 = np.zeros(n)

#x1 = x - y1
#x2 = x - y2
#x3 = x - y3
x4 = x - y4

#x1 = x1 + 0.025*np.random.randn(n)
#x2 = x2 + 0.1*np.random.randn(n)
#x3 = x3 + 0.05*np.random.randn(n)
x4 = x4 + 0.05*np.random.randn(n)

#pyplot.plot(x, x1, '.', color="blue")
#pyplot.plot(x, x2, '.', color="green")
#pyplot.plot(x, x3,'.', color="black")
pyplot.plot(x[350:], x4[350:], '.', color="red")
pyplot.xticks(np.arange(-10,45,5))
pyplot.legend(['QPSK', '16QAM', '64QAM', '256QAM'])
pyplot.show()