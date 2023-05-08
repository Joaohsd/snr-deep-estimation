import numpy as np
import pandas
from matplotlib import pyplot

# Preraring dataset
y1 = np.loadtxt('dataset/measured_qpsk.csv')
y2 = np.loadtxt('dataset/measured_16qam.csv')
y3 = np.loadtxt('dataset/measured_64qam.csv')
y4 = np.loadtxt('dataset/measured_256qam.csv')

n = len(y1)
# No offset
offset1 = y1[n-1] - 40
offset2 = y2[n-1] - 40
offset3 = y3[n-1] - 40
offset4 = y4[n-1] - 40

for i in range(n):
    y1[i] = y1[i] - offset1
    y2[i] = y2[i] - offset2
    y3[i] = y3[i] - offset3
    y4[i] = y4[i] - offset4

x = np.arange(-39.9,40.1,0.1)

x1 = np.zeros(n)
x2 = np.zeros(n)
x3 = np.zeros(n)
x4 = np.zeros(n)

x1 = x - y1
x2 = x - y2
x3 = x - y3
x4 = x - y4

x1 = x1 + 0.025*np.random.randn(n)
x2 = x2 + 0.1*np.random.randn(n)
x3 = x3 + 0.05*np.random.randn(n)
x4 = x4 + 0.05*np.random.randn(n)

np.savetxt('dataset/diff_qpsk.csv', x1)
np.savetxt('dataset/diff_16qam.csv', x2)
np.savetxt('dataset/diff_64qam.csv', x3)
np.savetxt('dataset/diff_256qam.csv', x4)

np.savetxt('dataset/adapted_qpsk.csv', y1)
np.savetxt('dataset/adapted_16qam.csv', y2)
np.savetxt('dataset/adapted_64qam.csv', y3)
np.savetxt('dataset/adapted_256qam.csv', y4)

pyplot.plot(x, x1, '.', color="blue")
pyplot.plot(x, x2, '.', color="green")
pyplot.plot(x, x3,'.', color="black")
pyplot.plot(x, x4, '.', color="red")
pyplot.xticks(np.arange(-40,45,5))
pyplot.legend(['QPSK', '16QAM', '64QAM', '256QAM'])
pyplot.show()
