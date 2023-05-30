import numpy as np
import pandas
from matplotlib import pyplot

# Preraring dataset
y1 = - np.loadtxt('measured_qpsk_80.csv')
y2 = - np.loadtxt('measured_16qam_80.csv')
y3 = - np.loadtxt('measured_64qam_80.csv')
y4 = - np.loadtxt('measured_256qam_80.csv')

n = len(y1)
# No offset
offset1 = y1[0] - 40
offset2 = y2[0] - 40
offset3 = y3[0] - 40
offset4 = y4[0] - 40

for i in range(n):
    y1[i] = y1[i] - offset1
    y2[i] = y2[i] - offset2
    y3[i] = y3[i] - offset3
    y4[i] = y4[i] - offset4

y1_inv = np.zeros(n)
y2_inv = np.zeros(n)
y3_inv = np.zeros(n)
y4_inv = np.zeros(n)

# Inversion
for i in range(n):
    y1_inv[i] = y1[n-1-i]
    y2_inv[i] = y2[n-1-i]
    y3_inv[i] = y3[n-1-i]
    y4_inv[i] = y4[n-1-i]

x = np.arange(-39.9,40.1,0.1)
x1 = np.zeros(n)
x2 = np.zeros(n)
x3 = np.zeros(n)
x4 = np.zeros(n)
x1 = x - y1_inv
x2 = x - y2_inv
x3 = x - y3_inv
x4 = x - y4_inv

np.savetxt('diff_qpsk.csv', x1)
np.savetxt('diff_16qam.csv', x2)
np.savetxt('diff_64qam.csv', x3)
np.savetxt('diff_256qam.csv', x4)

np.savetxt('adapted_qpsk.csv', y1_inv)
np.savetxt('adapted_16qam.csv', y2_inv)
np.savetxt('adapted_64qam.csv', y3_inv)
np.savetxt('adapted_256qam.csv', y4_inv)

pyplot.plot(x, x1, color="blue")
pyplot.show()
