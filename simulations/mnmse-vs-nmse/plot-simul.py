import numpy as np
import matplotlib.pyplot as plt

snr = np.arange(start=-39.9, stop=40.1, step=0.1)

nmse = np.loadtxt('nmse_sampling.csv')
adapt = np.loadtxt('adapt_sampling.csv')
adapt_old = np.loadtxt('adapt_sampling_old.csv')

nmse_l = nmse[299:]
adapt_l = adapt[299:]
snr_l = snr[299:]
adapt_old = adapt_old

print(nmse_l.shape)
print(adapt_l.shape)
print(snr_l.shape)
print(adapt_old.shape)


fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(111)

ax.plot(snr_l, snr_l,linestyle='solid',color='black')
ax.plot(snr_l, -1.0*nmse_l, linestyle=(0,(1,1)),color='red')
ax.plot(snr_l, adapt_l, linestyle=(0,(3,3)),color='green')
ax.plot(np.arange(start=-10, stop=40.25, step=0.25), adapt_old, linestyle = (0,(4,4)),color='blue')


ax.grid(visible=True)
ax.set_xlabel('SNR real do canal', fontsize=14)
ax.set_ylabel('Estimativa de SNR', fontsize=14)
ax.set_xticks(np.arange(-10,45,5))
ax.set_yticks(np.arange(-25,45,5))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.legend(['SNR Real','NMSE','Função NMSE Modificada','Função MSE Modificada'], fontsize=14)
'''
print(snr_l[80:89])
left, bottom, width, height = [0.6, 0.2, 0.25, 0.3]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(snr_l[80:89], snr_l[80:89], label='Real snr_l', linestyle=(0, (7,4)),color='red')
ax2.plot(snr_l[80:89], -1.0*nmse_l[80:89], label='nmse_l snr_l', linestyle=(0,(1,1)),color='black')
ax2.plot(snr_l[80:89], dnn_l[80:89], label='dnn_l snr_l', linestyle='solid',color='green')
ax2.grid(visible=True)
ax2.set_xticks(np.arange(20.0, 23.0, 1))
'''
plt.savefig('SimulMCSControl_10000samples_50hz_snrstep0.25_range40dB.png', dpi=400)
plt.show()
'''
plt.plot(snr_l, diff_nmse_l, color='red')
plt.plot(snr_l, diff_dnn_l, color='black')
plt.grid(visible=True)
plt.xticks(np.arange(10,36,2))
plt.legend(['MSE nmse_l','MSE dnn_l'])
plt.show()
'''
