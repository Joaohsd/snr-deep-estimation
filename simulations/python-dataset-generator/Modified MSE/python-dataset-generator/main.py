from simul_MSE import SnrAWGNChannel
import numpy as np
import matplotlib.pyplot as plt
import random

MSE_BLOCK_GNURADIO = 0
MSE_PAPER = 1
MSE_SIMPLE = 2

if __name__ == "__main__":
    # Number of samples to be generated
    N = 100000000

    # Modulation index (0: QPSK; 1: 16QAM; 2: 64QAM; 3: 256QAM)
    for modulation in range(4):
        # Object for the simulation
        snrSimulation = SnrAWGNChannel(N, modulation)

        # Execute the simulation with current MSE
        snrSimulation.execute(MSE_BLOCK_GNURADIO)
        # Plot tx symbols
        snrSimulation.plot_symbols(snrSimulation.tx_symbols, 'TX_Symbols')
        # Plot noise added to symbols with the last snr value
        snrSimulation.plot_symbols(snrSimulation.noise, 'AWGN')
        # Plot rx symbols with the last snr value
        snrSimulation.plot_symbols(snrSimulation.rx_symbols, 'RX_Symbols')
        # Plot MSE values
        snrSimulation.plot_mse('MSE_BLOCK_GNURADIO')
        # Plot diff MSE vs SNR
        snrSimulation.plot_diff_mse('DIFF_MSE_BLOCK_GNURADIO')
        print('END first')
        
        # Execute the simulation with MSE from paper
        snrSimulation.execute(MSE_PAPER)
        # Plot MSE values
        snrSimulation.plot_mse('MSE_PAPER')
        # Plot diff MSE vs SNR
        snrSimulation.plot_diff_mse('DIFF_MSE_PAPER')
        print('END second')

        # Execute the simulation with simple MSE
        snrSimulation.execute(MSE_SIMPLE)
        # Plot MSE values
        snrSimulation.plot_mse('MSE_SIMPLE')
        # Plot diff MSE vs SNR
        snrSimulation.plot_diff_mse('DIFF_MSE_SIMPLE')
        print('END third')



