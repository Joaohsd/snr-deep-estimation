import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import Sequential
from keras.models import model_from_json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Preraring dataset
X_l = np.loadtxt("adapted_64qam.csv")
y_l = np.loadtxt("diff_64qam.csv")
X = X_l[200:800]
y = y_l[200:800]

# Data Scaling from 0 to 1, X and y originally have very different scales.
X_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
y_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X_scaled = (X_scaler.fit_transform(X.reshape(-1,1)))
y_scaled = (y_scaler.fit_transform(y.reshape(-1,1)))

# load json and created model
json_file = open('model_64qam_144d.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_64qam_144d.h5")
print("Loaded model from disk")

# Predict the response variable with new data
predicted = loaded_model.predict(X_scaled)

x_graph = np.arange(-19.9, 40.1, 0.1)
# Plot in blue color the predicted adata and in green color the
# actual data to verify visually the accuracy of the model.
pyplot.plot(x_graph, y_scaler.inverse_transform(y_scaled), '.', markersize=4, color="black")
pyplot.plot(x_graph, y_scaler.inverse_transform(predicted), linewidth=2, color="red")
pyplot.grid()
pyplot.xlabel('Adapted Signal-to-Noise Ratio', fontsize=16)
pyplot.ylabel('DNN output', fontsize=16)
pyplot.xticks(np.arange(-20,45,5), fontsize=12)
pyplot.yticks(fontsize=12)
pyplot.legend(( 'Data', 'Predicted'), fontsize=14, loc='upper right')
pyplot.savefig('snr_64qam_dnn144.pdf')
pyplot.show()
