import numpy as np
import random
from qam import *
import matplotlib.pyplot as plt
from numba import jit

class SnrAWGNChannel:
    def __init__(self, N=10000, modulation=0):
        # Defining class attributes
        # Modulation index
        self.modulation = modulation

        # SNR values 
        self.snr_values = np.arange(-39.9,40.1,0.1)

        # Samples
        self.number_of_samples = N

        # Symbols
        self.noise = np.zeros(N, dtype=complex)
        self.tx_symbols = np.zeros(N, dtype=complex)
        self.rx_symbols = np.zeros(N, dtype=complex)

        # MSE history
        self.mse_values = np.zeros(len(self.snr_values), dtype=np.float64)
        
        # Events history
        self.external_events = np.zeros(len(self.snr_values), dtype=np.float64)

        # Verify desired modulation
        if self.modulation == 0:
            self.max_value = np.float32(2.0)
            self.factor = np.float32(DENORM_QPSK)       # sqrt(2)
            self.beta = np.float32(1.0)
            self.modulation_nbits = MOD_QPSK_NBITS
            print ("***** config QPSK *****")
        elif self.modulation == 1:
            self.max_value = np.float32(4.0)
            self.factor = np.float32(DENORM_16QAM)        # sqrt(10)     
            self.beta = np.float32(3.0)
            self.modulation_nbits = MOD_16QAM_NBITS
            print ("***** config 16QAM *****")
        elif self.modulation == 2:
            self.max_value = np.float32(8.0)
            self.factor = np.float32(DENORM_64QAM)         # sqrt(42)
            self.beta = np.float32(7.0)
            self.modulation_nbits = MOD_64QAM_NBITS
            print ("***** config 64QAM *****")
        else:
            self.max_value = np.float32(16.0)
            self.factor = np.float32(DENORM_256QAM)         # sqrt(170)
            self.beta = np.float32(15.0)
            self.modulation_nbits = MOD_256QAM_NBITS
            print ("***** config 256QAM *****")

    @jit 
    def _mse_calc(self, N, complex_input):
        max_value = self.max_value
        factor = self.factor
        beta = self.beta
        mse = np.float32(0.0)
        for i in range(N):
            d_real = np.abs(np.real(complex_input[i]))*factor
            d_imag = np.abs(np.imag(complex_input[i]))*factor

            dec_real = np.round((d_real + 1.0)/2.0)*2.0 - 1.0
            comp_real = np.float32(0.0)
            if (d_real > max_value):
                    dec_real = max_value - 1.0
                    comp_real = np.abs(d_real - dec_real)
            dec_imag = np.round((d_imag + 1.0)/2.0)*2.0 - 1.0
            comp_imag = np.float32(0.0)
            if (d_imag > max_value):
                    dec_imag = max_value - 1.0
                    comp_imag = np.abs(d_imag - dec_imag)
            comp_real = beta*comp_real/factor
            comp_imag = beta*comp_imag/factor
            d_real = (d_real - dec_real)/factor
            d_imag = (d_imag - dec_imag)/factor
            mse = mse + d_real * d_real + d_imag * d_imag + comp_real * comp_real + comp_imag * comp_imag
        return -10*np.log10(mse/N)
    

    @jit 
    def _mse_calc_paper(self, N, complex_input):
        max_value = self.max_value
        factor = self.factor
        beta = self.beta
        mse = np.float32(0.0)
        energy = np.float32(0.0)
        external_events = 0
        for i in range(N):
            d_real = np.abs(np.real(complex_input[i]))*factor
            d_imag = np.abs(np.imag(complex_input[i]))*factor

            dec_real = np.floor((d_real)/2.0)*2.0 + 1.0
            comp_real = np.float32(0.0)
            if (d_real > max_value):
                dec_real = max_value - 1.0
                erro = d_real - np.abs(dec_real)
                b = 2 * (d_real - max_value)    # Componente b
                a = erro - b    # Estimativa do erro que já foi computado
                comp_real = erro*erro + beta * (2*a*b + b*b)
            else:
                comp_real = d_real - np.abs(dec_real)
                comp_real = comp_real * comp_real

            dec_imag = np.floor((d_imag)/2.0)*2.0 + 1.0
            comp_imag = np.float32(0.0)
            if (d_imag > max_value):
                dec_imag = max_value - 1.0
                erro = d_imag - np.abs(dec_imag)
                b = 2 * (d_imag - max_value)    # Componente b
                a = erro - b     # Estimativa do erro que já foi computado
                comp_imag = erro*erro + beta * (2*a*b + b*b)
            else:
                comp_imag = d_imag - np.abs(dec_imag)
                comp_imag = comp_imag * comp_imag
            
            if (d_real > max_value) or (d_imag > max_value):
                external_events += 1

            mse = mse + comp_real + comp_imag
            energy = energy + dec_real*dec_real + dec_imag*dec_imag
            
        return -10*np.log10(mse/energy), 100.0*np.float64(external_events)/N
    
    '''
    @jit 
    def _mse_calc_paper(self, N, complex_input):
        max_value = self.max_value
        factor = self.factor
        beta = self.beta
        mse = np.float32(0.0)
        energy = np.float32(0.0)
        external_events = 0
        for i in range(N):
            d_real = np.abs(np.real(complex_input[i]))*factor
            d_imag = np.abs(np.imag(complex_input[i]))*factor

            dec_real = np.floor((d_real)/2.0)*2.0 + 1.0
            comp_real = np.float32(0.0)

            dec_imag = np.floor((d_imag)/2.0)*2.0 + 1.0
            comp_imag = np.float32(0.0)

            if d_real > max_value:
                external_events = external_events + 1
                dec_real = max_value - 1.0
                
                if d_imag > max_value:
                    dec_imag = max_value - 1.0
                
                secondNearestNeighboor = dec_real + dec_imag*1j
                min_distance = np.inf

                for i in range(3):
                    new_imag = dec_imag
                    new_real = dec_real
                    if i == 0: # Up
                        if dec_imag != (max_value - 1.0):
                            new_imag = dec_imag + 2.0
                            distance = np.sqrt(np.power(d_real - dec_real, 2) + np.power(d_imag - new_imag, 2))
                    elif i == 1: # Left
                        new_real = dec_real - 2.0
                        distance = np.sqrt(np.power(d_real - new_real, 2) + np.power(d_imag - dec_imag, 2))
                    elif i == 2: # Down
                        new_imag = dec_imag - 2.0
                        distance = np.sqrt(np.power(d_real - dec_real, 2) + np.power(d_imag - new_imag, 2))

                    if distance < min_distance:
                        min_distance = distance
                        secondNearestNeighboor = new_real + new_imag*1j
                
                dec_real = np.real(secondNearestNeighboor)
                dec_imag = np.imag(secondNearestNeighboor)

                comp_real = np.abs(d_real - dec_real)
                comp_real = comp_real * comp_real

                comp_imag = np.abs(d_imag - dec_imag)
                comp_imag = comp_imag * comp_imag
        

            elif d_imag > max_value:
                external_events = external_events + 1
                dec_imag = max_value - 1.0
                
                secondNearestNeighboor = dec_real + dec_imag*1j
                min_distance = np.inf

                for i in range(3):
                    new_imag = dec_imag
                    new_real = dec_real
                    if i == 0: # Right
                        if dec_real != (max_value - 1.0):
                            new_real = dec_real + 2.0
                            distance = np.sqrt(np.power(d_real - new_real, 2) + np.power(d_imag - dec_imag, 2))
                    elif i == 1: # Left
                        new_real = dec_real - 2.0
                        distance = np.sqrt(np.power(d_real - new_real, 2) + np.power(d_imag - dec_imag, 2))
                    elif i == 2: # Down
                        new_imag = dec_imag - 2.0
                        distance = np.sqrt(np.power(d_real - dec_real, 2) + np.power(d_imag - new_imag, 2))

                    if distance < min_distance:
                        min_distance = distance
                        secondNearestNeighboor = new_real + new_imag*1j

                dec_real = np.real(secondNearestNeighboor)
                dec_imag = np.imag(secondNearestNeighboor)

                comp_real = np.abs(d_real - dec_real)
                comp_real = comp_real * comp_real

                comp_imag = np.abs(d_imag - dec_imag)
                comp_imag = comp_imag * comp_imag
            
            else:
                comp_real = np.abs(d_real - dec_real)
                comp_real = comp_real * comp_real

                comp_imag = np.abs(d_imag - dec_imag)
                comp_imag = comp_imag * comp_imag
                
            mse = mse + comp_real + comp_imag
            energy = energy + dec_real*dec_real + dec_imag*dec_imag
            
        return -10*np.log10(mse/energy), 100.0*np.float64(external_events)/N
    '''
    @jit
    def _mse_calc_simple(self, N, complex_input):
        max_value = self.max_value
        factor = self.factor
        mse = np.float32(0.0)
        for i in range(N):
            d_real = np.abs(np.real(complex_input[i]))*factor
            d_imag = np.abs(np.imag(complex_input[i]))*factor

            dec_real = np.round((d_real)/2.0)*2.0 - 1.0
            comp_real = np.float32(0.0)
            if (d_real > max_value):
                dec_real = max_value - 1.0
                comp_real = np.abs(d_real - dec_real)
                comp_real = comp_real * comp_real
            else:
                comp_real = np.abs(d_real - dec_real)
                comp_real = comp_real * comp_real

            dec_imag = np.round((d_imag)/2.0)*2.0 - 1.0
            comp_imag = np.float32(0.0)
            if (d_imag > max_value):
                dec_imag = max_value - 1.0
                comp_imag = np.abs(d_imag - dec_imag)
                comp_imag = comp_imag * comp_imag
            else:
                comp_imag = np.abs(d_imag - dec_imag)
                comp_imag = comp_imag * comp_imag

            mse = mse + comp_real + comp_imag

        return -10*np.log10(mse/N)

    def _generate_qam_symbols(self, N=10000, modulation=0):
        # Initialize the arrays
        self.tx_symbols = np.zeros(N, dtype=complex)
        imag = np.zeros(N, dtype=np.float64)
        real = np.zeros(N, dtype=np.float64)
        # Order of modulation 
        modulation_order = pow(2, self.modulation_nbits)
        # Create array of N decimal values
        decimal_values = np.random.randint(0, modulation_order, size=N)
        # Create a mask to map QAM sumbol
        mask = 0xFF >> (8 - int(self.modulation_nbits/2.0))
        # Get imag part
        imag = np.array(qam_map[modulation])[(decimal_values & mask)]
        # Get real part
        real = np.array(qam_map[modulation])[(decimal_values >> int(self.modulation_nbits/2.0)) & mask]
        # Get symbols by the sum of real part and imaginary part
        self.tx_symbols = real+imag*1j
        print('Symbols have been generated')

    def _generate_noise(self, snr, N=10000):
        # Calculate Gain for the SNR
        noise_gain = pow(10,(-snr/20.0))
        # Amplitude for the desired SNR
        noise_amplitude = 0.8971686730526076 * noise_gain
        # Generate noise
        self.noise = noise_amplitude * (np.random.randn(N) + np.random.randn(N)*1j)
        print('Noise has been generated')

    def _transmit_symbols(self, snr):
        # Symbols
        self._generate_qam_symbols(self.number_of_samples, self.modulation)
        # Noise
        self._generate_noise(snr, self.number_of_samples)
        print('Symbols have been transmitted')
    
    def _receive_symbols(self):
        # Symbols
        self.rx_symbols = self.tx_symbols + self.noise
        print('Symbols have been received')

    def plot_symbols(self, symbols, title):
        modulation_order = pow(2, self.modulation_nbits)
        fig = plt.figure(figsize=(8,8))
        plt.plot(np.real(symbols), np.imag(symbols), '.')
        plt.xlabel('Q')
        plt.ylabel('I')
        plt.title(title)
        plt.savefig('img/'+ str(modulation_order) + '-QAM' + '/' + title + '.png', dpi=400)
        #plt.show()
        
    def plot_mse(self, title):
        modulation_order = pow(2, self.modulation_nbits)
        fig = plt.figure(figsize=(8,8))
        plt.plot(self.snr_values, self.mse_values, '.', color='black')
        plt.plot(self.snr_values, self.snr_values, color='red')
        plt.xlabel('SNR')
        plt.ylabel('MSE')
        plt.xlim((40,-40))
        plt.ylim((max(self.mse_values),min(self.mse_values)))
        plt.legend(['MSE', 'Real SNR'])
        plt.title(title)
        plt.savefig('img/'+ str(modulation_order) + '-QAM' + '/' + title + '.png', dpi=400)
        #plt.show()

    def _modulation_to_string(self, modulation):
         if modulation == 0:
              return "qpsk"
         elif modulation == 1:
              return "16qam"
         elif modulation == 2:
              return "64qam"
         elif modulation == 3:
              return "256qam"

    def plot_diff_mse(self, title):
        modulation_order = pow(2, self.modulation_nbits)
        fig = plt.figure(figsize=(8,8))
        plt.plot(self.snr_values, (self.snr_values - self.mse_values), '.')
        plt.xlabel('SNR')
        plt.ylabel('MSE')
        plt.xlim((40,-40))
        plt.ylim((-20,20))
        plt.title(title)
        plt.savefig('img/'+ str(modulation_order) + '-QAM' + '/' + title + '.png', dpi=400)
        #plt.show()
    
    def execute(self, mse_option):
        index = 0
        for snr in self.snr_values:
            # Transmit Symbols with desired SNR
            self._transmit_symbols(snr)
            # Receive Symbols
            self._receive_symbols()
            # MSE
            if mse_option == 0:
                # Calculate MSE Modified
                self.mse_values[index] = self._mse_calc(self.number_of_samples, self.rx_symbols)
            elif mse_option == 1:
                # Calculate MSE Paper
                self.mse_values[index], self.external_events[index] = self._mse_calc_paper(self.number_of_samples, self.rx_symbols)
            else:
                # Calculate MSE Simple
                self.mse_values[index] = self._mse_calc_simple(self.number_of_samples, self.rx_symbols)
            index = index + 1
        
        with open("dataset/measured_"+self._modulation_to_string(self.modulation)+".csv", "a+") as measured:
            for mse in self.mse_values:
                measured.write(str(mse)+'\n')
        
        with open("dataset/events_"+self._modulation_to_string(self.modulation)+".csv", "a+") as events:
            for event in self.external_events:
                events.write(str(event)+'\n')
        

