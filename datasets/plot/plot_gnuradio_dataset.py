import matplotlib.pyplot as plt
import numpy as np

measured = np.loadtxt('measured_256qam.csv')
diff = np.loadtxt('diff_256qam.csv')
measured = measured[300:]
diff = diff[300:]

snr = np.arange(-39.9, 40.1, 0.1)
snr = snr[300:]
print(snr)

plt.plot(snr, diff, '.', color='black')
plt.plot(snr, measured, '.', color='red')
plt.xlabel('Real SNR')
plt.ylabel('DIFF SNR')
plt.xticks(np.arange(-10, 40, 5))
plt.show()