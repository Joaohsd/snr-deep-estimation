import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import Sequential
import keras
from datetime import datetime
from packaging import version

# Preraring dataset
X_l = np.loadtxt("dataset/adapted_256qam.csv")
y_l = np.loadtxt("dataset/diff_256qam.csv")
logdir = "logs/scalars/" + "256QAM"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

X = X_l[200:800]
y = y_l[200:800]

# Data Scaling from 0 to 1, X and y originally have very different scales.
X_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
y_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X_scaled = (X_scaler.fit_transform(X.reshape(-1,1)))
y_scaled = (y_scaler.fit_transform(y.reshape(-1,1)))

# New sequential network structure.
model = Sequential()

# Input layer with dimension 1 and hidden layer i with 6 neurons. 
model.add(Dense(1, kernel_initializer='uniform', activation='linear'))
# Hidden layer j with 4 neurons plus activation layer.
model.add(Dense(4, activation='linear'))
# Hidden layer k with 4 neurons.
model.add(Dense(4, activation='sigmoid'))
# Output Layer.
model.add(Dense(1))

# Model is derived and compiled using mean square error as loss
# function, accuracy as metric and gradient descent optimizer.
model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])

# Training model with train data. Fixed random seed:
#np.random.seed(123)
model.fit(X_scaled, y_scaled, epochs=2000, batch_size=2, callbacks=[tensorboard_callback], verbose=2)

# Serialize model to JSON
model_json = model.to_json()
with open("model_256qam_144d.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_256qam_144d.h5")
print("Saved model to disk")

# Predict the response variable with new data
predicted = model.predict(X_scaled)

# Plot in blue color the predicted adata and in green color the
# actual data to verify visually the accuracy of the model.
pyplot.plot(y_scaler.inverse_transform(predicted), color="red")
pyplot.plot(y_scaler.inverse_transform(y_scaled), color="green")
pyplot.legend(('Predicted 256QAM', 'Data'), loc='lower right')
pyplot.show()
