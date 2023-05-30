"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr

import pmt

class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block example - a simple multiply const"""

    def __init__(self):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='Multiply Controlled Block',   # will show up in GRC
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
        self.snr = 0

        self.message_port_register_in(pmt.intern('snr'))
        self.set_msg_handler(pmt.intern('snr'), self.handle_msg)

    def handle_msg(self, msg_pmt):
        snr = pmt.intern("set_snr")
        #print('[MULTIPLY BLOCK] KEY: ',pmt.car(pmt.car(msg_pmt)))
        #print('[MULTIPLY BLOCK] VALUE: ',pmt.cdr(pmt.car(msg_pmt)))
        if pmt.car(pmt.car(msg_pmt)) == snr:
            #print('[MULTIPLY BLOCK] SNR RECEIVED')
            msg = pmt.cdr(pmt.car(msg_pmt))
            #print('[MULTIPLY BLOCK] SNR:',msg)
            if self.snr != pmt.to_python(msg):
                self.snr = pmt.to_python(msg)

    def work(self, input_items, output_items):
        """example: multiply with constant"""
        output_items[0][:] = input_items[0] * pow(10,((-1.0*self.snr)/20.0))
        return len(output_items[0])