import numpy as np
import matplotlib.pyplot as plt

snr = np.arange(-10, 40.25, 0.25)

nmse = np.loadtxt('nmse_sampling.csv')
dnn = np.loadtxt('dnn_sampling.csv')
adapt = np.loadtxt('adapt_sampling.csv')

nmse_l = nmse[40:]
dnn_l = dnn[40:]
adapt_l = adapt[40:]
snr_l = snr[40:]


diff_nmse_l = pow(-1.0*nmse_l-snr_l,2)
diff_dnn_l = pow(dnn_l-snr_l,2)


fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(111)

ax.plot(snr_l, -1.0*nmse_l, linestyle=(0,(1,1)),color='black')
ax.plot(snr_l, snr_l,linestyle=(0, (7,4)),color='red')
ax.plot(snr_l, dnn_l, linestyle='solid',color='green')
ax.plot(snr_l, adapt_l, linestyle=(0, (3,3)),color='blue')

nmse_f = nmse[40::6]
dnn_f = dnn[40::6]
adapt_f = adapt[40::6]
snr_f = snr[40::6]

'''
ax.plot(snr_f, -1.0*nmse_f, linestyle=(0,(1,1)), marker='o', markersize=4,color='black')
ax.plot(snr_f, dnn_f, linestyle='solid', marker='+', markersize=7,color='green')
ax.plot(snr_f, adapt_f, linestyle=(0, (3,3)), marker='s', markersize=4,color='blue')
'''

ax.grid(visible=True)
ax.set_xlabel('SNR real do canal', fontsize=14)
ax.set_ylabel('Estimativa de SNR', fontsize=14)
ax.set_xticks(np.arange(0,42,2))
ax.set_yticks(np.arange(-10,42,2))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.legend(['NMSE SNR','REAL SNR','DNN SNR', 'MMSE SNR'], fontsize=14)
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
plt.savefig('Simul256QAM_10000samples_50hz_snrstep0.25_range40dB.png', dpi=400)
plt.show()
'''
plt.plot(snr_l, diff_nmse_l, color='red')
plt.plot(snr_l, diff_dnn_l, color='black')
plt.grid(visible=True)
plt.xticks(np.arange(10,36,2))
plt.legend(['MSE nmse_l','MSE dnn_l'])
plt.show()

print('DIFF nmse_l:',np.sum(diff_nmse_l)/len(diff_nmse_l))
print('DIFF dnn_l:',np.sum(diff_dnn_l)/len(diff_dnn_l))
'''