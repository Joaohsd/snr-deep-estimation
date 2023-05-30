import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import Sequential
from keras.models import model_from_json
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Preraring dataset
X_l = np.loadtxt("adapted_qpsk.csv")
y_l = np.loadtxt("diff_qpsk.csv")
X = X_l[200:800]
y = y_l[200:800]

# Data Scaling from 0 to 1, X and y originally have very different scales.
X_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
y_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X_scaled = (X_scaler.fit_transform(X.reshape(-1,1)))
y_scaled = (y_scaler.fit_transform(y.reshape(-1,1)))

# load json and created model
json_file = open('model_qpsk_144d.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_qpsk_144d.h5")


# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

original_stdout = sys.stdout # Save a reference to the original standard output
with open('qpsk_report.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    for layer in loaded_model.layers:
        g=layer.get_config()
        h=layer.get_weights()
        print ()
        print (g,'\n')
        print (h,'\n')
#    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
sys.stdout = original_stdout # Reset the standard output to its original value