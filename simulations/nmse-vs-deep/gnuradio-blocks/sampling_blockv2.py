"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr
import pmt
import time

class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block - Sampling DNN and NMSE"""

    def __init__(self, samples=5000, freq_dnn_sampling=20):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='Sampling NMSE and DNN Output',   # will show up in GRC
            in_sig=[np.float32, np.float32],
            out_sig=[np.float32, np.float32]
        )
        # Number of samples per value
        self.n_samples = samples
        self.nmse_nsamples = 0
        self.dnn_nsamples = 0
        self.adapt_nsamples = 0
        self.time_to_sample_dnn = (1/freq_dnn_sampling)*1000

        # Flag to get samples provided by the inputs
        self.get_samples = False
        self.change_snr = True

        # SNR values
        self.snr = np.arange(start=-40.0, stop=40.25, step=0.25)
        self.snr_index = 0

        # Samples for NMSE Input
        self.nmse_samples = np.zeros(self.n_samples, dtype=np.float32)

        # Samples for DNN Input
        self.dnn_samples = np.zeros(self.n_samples, dtype=np.float32)

        # Samples for SNR Adapt Input
        self.adapt_samples = np.zeros(self.n_samples, dtype=np.float32)

        # Last time sample
        self.last_time_snr_changed = int(time.time() * 1000)
        self.last_time_sample_dnn = int(time.time() * 1000)

        self.message_port_register_in(pmt.intern('nmse'))
        self.set_msg_handler(pmt.intern('nmse'), self.handle_msg)
        self.message_port_register_out(pmt.intern('snr'))
        self.key = pmt.intern("set_snr")

    def handle_msg(self, msg_pmt):
        nmse = pmt.intern("NMSE")
        if pmt.car(msg_pmt) == nmse:
            msg = pmt.cdr(msg_pmt)
            #print(msg)
            if self.get_samples and self.nmse_nsamples < self.n_samples:
                #print('[SAMPLING BLOCK] SAMPLING NMSE!')
                self.nmse_samples[self.nmse_nsamples] = pmt.to_python(msg)
                self.nmse_nsamples += 1
    	
    def work(self, input_items, output_items):
        output_items[0][:] = input_items[0][:]
        output_items[1][:] = input_items[1][:]

        if self.change_snr and self.snr_index < self.snr.shape[0]:
            print('[SAMPLING BLOCK] SNR CHANGED!')
            val = pmt.from_float(self.snr[self.snr_index])
            snr_dict = pmt.make_dict()
            snr_dict = pmt.dict_add(snr_dict, self.key, val)
            self.message_port_pub(pmt.intern('snr'), snr_dict)
            self.last_time_snr_changed = int(time.time() * 1000)   
            self.snr_index += 1
            self.change_snr = False

        if (not self.change_snr) and (not self.get_samples):
            self.now = int(time.time() * 1000)
            if (self.now - self.last_time_snr_changed) >= 5000 :
                self.get_samples = True

        if self.get_samples:
            self.now = int(time.time() * 1000)
            if (self.now - self.last_time_sample_dnn) > self.time_to_sample_dnn:
                if self.dnn_nsamples < self.n_samples:
                    self.dnn_samples[self.dnn_nsamples] = np.mean(input_items[0])
                    self.adapt_samples[self.adapt_nsamples] = np.mean(input_items[1])
                    self.dnn_nsamples += 1
                    self.adapt_nsamples += 1
                    self.last_time_sample_dnn = int(time.time() * 1000)
                elif self.nmse_nsamples >= self.n_samples:
                    with open("dnn_sampling.csv", "a+") as dnn_file:
                        dnn_file.write(str(np.mean(self.dnn_samples))+'\n')
                    with open("nmse_sampling.csv", "a+") as nmse_file:
                        nmse_file.write(str(np.mean(self.nmse_samples))+'\n')
                    with open("adapt_sampling.csv", "a+") as adapt_file:
                        adapt_file.write(str(np.mean(self.adapt_samples))+'\n')
                    self.get_samples = False
                    self.change_snr = True
                    self.nmse_nsamples = 0
                    self.dnn_nsamples = 0
                    self.adapt_nsamples = 0
                    self.dnn_samples[:] = 0
                    self.nmse_samples[:] = 0
                    self.adapt_samples[:] = 0
                    print('[SAMPLING BLOCK] Sampling finished for ', self.snr[self.snr_index - 1], '!')

        return len(output_items[0])