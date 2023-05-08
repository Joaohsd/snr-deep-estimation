import pmt
import numpy as np
import tensorflow as tf
from gnuradio import gr
from tensorflow.python.ops.numpy_ops import np_config
from sklearn import preprocessing
from keras.models import model_from_json
from numba import jit

class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    def __init__(self, dnn_dis=0.0):  # DNN disable parameter
        gr.sync_block.__init__(
            self,
            name='SINR Estimation-DNN',   # will show up in GRC
            in_sig=[np.complex64],
            out_sig=[np.float32, np.float32, np.float32]
        )

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
                try:
    			# Currently, memory growth needs to be the same across GPUs
                        for gpu in gpus:
                                tf.config.experimental.set_memory_growth(gpu, True)
                        logical_gpus = tf.config.list_logical_devices('GPU')
                        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                        # Memory growth must be set before GPUs have been initialized
                        print(e)
        self.mcs = 0
        self.mcs_d = 0
        self.mcs_control = 0
        self.mcs_control_d = 0
        self.modul = 0
        self.modul_d = 0
        self.message_port_register_in(pmt.intern('msg_in'))
        self.set_msg_handler(pmt.intern('msg_in'), self.handle_msg)
        self.message_port_register_out(pmt.intern('mcs'))
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
        self.dnn_dis = dnn_dis
        self.key = pmt.intern("set_tx_mcs")
        self.mcs_table = (5.5 + 1.5, 6.5 + 1.5, 7.0 + 1.5, 8.0 + 1.5, 12.0 + 1.5, 13.0 + 1.5, 14.0 + 1.5,\
		14.5 + 1.5, 18.5 + 1.5, 19.5 + 1.5, 20.0 + 1.5, 25.5 + 1.5, 26.5 + 1.5, 27.5 + 1.5)
        self.mcs_max = 13
        self.mse = 20

        # Preraring dataset
        X_l = np.loadtxt("adapted_qpsk.csv")
        y_l = np.loadtxt("diff_qpsk.csv")
        X = X_l[200:800]
        y = y_l[200:800]

        self.X_scaler_qpsk = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.y_scaler_qpsk = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.X_scaler_qpsk.fit(X.reshape(-1,1))
        self.y_scaler_qpsk.fit(y.reshape(-1,1))

        X_l = np.loadtxt("adapted_16qam.csv")
        y_l = np.loadtxt("diff_16qam.csv")
        X = X_l[200:800]
        y = y_l[200:800]

        self.X_scaler_16 = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.y_scaler_16 = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.X_scaler_16.fit(X.reshape(-1,1))
        self.y_scaler_16.fit(y.reshape(-1,1))

        X_l = np.loadtxt("adapted_64qam.csv")
        y_l = np.loadtxt("diff_64qam.csv")
        X = X_l[200:800]
        y = y_l[200:800]

        self.X_scaler_64 = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.y_scaler_64 = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.X_scaler_64.fit(X.reshape(-1,1))
        self.y_scaler_64.fit(y.reshape(-1,1))

        X_l = np.loadtxt("adapted_256qam.csv")
        y_l = np.loadtxt("diff_256qam.csv")
        X = X_l[200:800]
        y = y_l[200:800]

        self.X_scaler_256 = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.y_scaler_256 = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.X_scaler_256.fit(X.reshape(-1,1))
        self.y_scaler_256.fit(y.reshape(-1,1))

        # load json and created models
        self.json_file = open('model_qpsk_144d.json', 'r')
        self.model_qpsk_json = self.json_file.read()
        self.json_file.close()
        self.model_qpsk = model_from_json(self.model_qpsk_json)
        # load weights into new model
        self.model_qpsk.load_weights("model_qpsk_144d.h5")

        # load json and created models
        self.json_file = open('model_16qam_144d.json', 'r')
        self.model_16_json = self.json_file.read()
        self.json_file.close()
        self.model_16 = model_from_json(self.model_16_json)
        # load weights into new model
        self.model_16.load_weights("model_16qam_144d.h5")

        # load json and created models
        self.json_file = open('model_64qam_144d.json', 'r')
        self.model_64_json = self.json_file.read()
        self.json_file.close()
        self.model_64 = model_from_json(self.model_64_json)
        # load weights into new model
        self.model_64.load_weights("model_64qam_144d.h5")

        # load json and created models
        self.json_file = open('model_256qam_144d.json', 'r')
        self.model_256_json = self.json_file.read()
        self.json_file.close()
        self.model_256 = model_from_json(self.model_256_json)
        # load weights into new model
        self.model_256.load_weights("model_256qam_144d.h5")

        # Default configuration
        self.X_scaler = self.X_scaler_qpsk
        self.y_scaler = self.y_scaler_qpsk
        self.model = self.model_qpsk
        self.max_value = np.float32(2.0)
        self.factor = np.float32(1.414213562)
        self.beta = np.float32(1.0)

    def handle_msg(self, msg_pmt):
        set_rx_mcs = pmt.intern("MCS")
        if (pmt.car(msg_pmt) == set_rx_mcs):
                msg = pmt.cdr(msg_pmt)
                print (msg)
                self.mcs_d = self.mcs
                self.mcs = pmt.to_python(msg)
        if self.mcs == self.mcs_d:
                return
        if self.mcs < 4:
                self.modul_d = self.modul
                self.modul = 0
        elif self.mcs < 8:
                self.modul_d = self.modul
                self.modul = 1
        elif self.mcs < 11:
                self.modul_d = self.modul
                self.modul = 2
        elif self.mcs < 14:
                self.modul_d = self.modul
                self.modul = 3
        if self.modul == self.modul_d:
                return
        if self.modul == 0:
                self.X_scaler = self.X_scaler_qpsk
                self.y_scaler = self.y_scaler_qpsk
                self.model = self.model_qpsk
                self.max_value = np.float32(2.0)
                self.factor = np.float32(1.414213562)
                self.beta = np.float32(1.0)
                print(self.calc_error(18.5))
                print ("***** config DNN: QPSK *****")
        elif self.modul == 1:
                self.X_scaler = self.X_scaler_16
                self.y_scaler = self.y_scaler_16
                self.model = self.model_16
                self.max_value = np.float32(4.0)
                self.factor = np.float32(3.16227766)
                self.beta = np.float32(2.0)
                print(self.calc_error(18.5))
                print ("***** config DNN: 16QAM *****")
        elif self.modul == 2:
                self.X_scaler = self.X_scaler_64
                self.y_scaler = self.y_scaler_64
                self.model = self.model_64
                self.max_value = np.float32(8.0)
                self.factor = np.float32(6.4807407)
                self.beta = np.float32(3.0)
                print(self.calc_error(18.5))
                print ("***** config DNN: 64QAM *****")
        elif self.modul == 3:
                self.X_scaler = self.X_scaler_256
                self.y_scaler = self.y_scaler_256
                self.model = self.model_256
                self.max_value = np.float32(16.0)
                self.factor = np.float32(13.038413)
                self.beta = np.float32(4.0)
                print(self.calc_error(18.5))
                print ("***** config DNN: 256QAM *****")
    @jit
    def mse_calc(self, n, complex_input):
        max_value = self.max_value
        factor = self.factor
        beta = self.beta
        mse = np.float32(0.0)
        for i in range(n):
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
        return -10.0*np.log10(mse/n)

    @tf.function(jit_compile=True)
    def predict(self, x):
        return self.model(x)

#    @jit
    def calc_error(self, snr):
        x = np.arange(1)
        x[0] = (snr)
        y = self.X_scaler.transform(x.reshape(-1, 1))
        pred = self.predict(y)
        scaled = self.y_scaler.inverse_transform(pred)
        return scaled

    def work(self, input_items, output_items):
        n = np.size(input_items[0])
        snr_adapt = self.mse_calc(n, input_items[0][:])
        error = self.calc_error(snr_adapt)
        dnn_comp = error + snr_adapt
        self.mse += dnn_comp - self.mse/64
        mse_f = self.mse/64
        if self.mcs_control > 0:				# fallback test
                if mse_f < self.mcs_table[self.mcs_control]:
                        self.mcs_control -= 1
        if self.mcs_control < (self.mcs_max):			# fall forward test
                if mse_f > (self.mcs_table[self.mcs_control+1] + 1.0):
                        self.mcs_control += 1
        if self.mcs_control_d != self.mcs_control:
                self.mcs_control_d = self.mcs_control
                val = pmt.from_long(self.mcs_control)
                mcs_dict = pmt.make_dict()
                mcs_dict = pmt.dict_add(mcs_dict, self.key, val)
                self.message_port_pub(pmt.intern('mcs'), mcs_dict)
        output_items[0][:] = snr_adapt
        output_items[1][:] = self.calc_error(18.5)
        output_items[2][:] = mse_f
        return len(output_items[0])

