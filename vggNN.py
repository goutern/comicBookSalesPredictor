import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.backend import sigmoid
import keras.backend as K
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Activation, BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout, Flatten
from keras.models import Sequential
from keras.utils.generic_utils import get_custom_objects
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.stats import stats

# Importing sklearn libraries

epochs = 1000
batch_size = 8

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish': Activation(swish)})


def tilted_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True



def plot_loss(history):
    start_epoch = 2
    fig = plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'][start_epoch:])
    plt.plot(history.history['val_loss'][start_epoch:])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


# Save memory just in case
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    except RuntimeError as e:
        print(e)


#Load up the clean data
clean_data = np.load('cleanData.npy',allow_pickle=True)

data_y = []
data = clean_data


print("Loading Data Set")

features = np.load("features.npy")
data_y = np.load("data_y.npy")


#Remove outliers
z_threshold = 1
print("Removing Outliers")
z = np.abs(stats.zscore(data[:,6].astype(int)))

# print(np.where(z > threshold))
print(data.shape)
print(data_y.shape)

data = data[(z < z_threshold)]
data_y = data_y[(z < z_threshold)]

print("Outliers Removed")
print(data.shape)
print(data_y.shape)

#train test validation split
np.random.rand(42)

indices = np.random.permutation(data.shape[0])
index = int(0.8 * len(indices))
percent = int(0.1 * len(indices))

training_idx, test_idx, val_idx = indices[:index], indices[index:index+percent], indices[index+percent:]
train_data, test_data, val_data = data[training_idx,:], data[test_idx,:], data[val_idx,:]

print(features.shape)
train, test, val = features[training_idx], features[test_idx], features[val_idx]
train_y, test_y, val_y = data_y[training_idx], data_y[test_idx], data_y[val_idx]

# Creating a checkpointer
checkpointer = ModelCheckpoint(filepath='scratchmodel.best.hdf5',
                               verbose=1, save_best_only=True)

callbacks = [
    EarlyStoppingByLossVal(monitor='val_loss', value=0.001, verbose=1),
    checkpointer]


# Base Line model
def baseline_model():
    model_scratch = Sequential()
    model_scratch.add(Flatten())
    model_scratch.add(Dense(512, activation=swish, input_shape=train.shape[1:]))
    model_scratch.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model_scratch.add(Dropout(0.5))
    model_scratch.add(Dense(512, activation=swish))
    model_scratch.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model_scratch.add(Dropout(0.5))
    model_scratch.add(Dense(1, activation=swish))
    model_scratch.compile(loss=lambda y, f: tilted_loss(quantile, y, f), optimizer='adagrad',
                          metrics=['mse', 'mae', 'mape'])
    return model_scratch

quantile = 0.9
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size= 100, verbose=False)

history = estimator.fit(train, train_y, batch_size=batch_size, epochs=epochs,
          validation_data=(val, val_y), callbacks = callbacks,
          verbose=1, shuffle=True)

prediction = estimator.predict(test)


# Display graphs and metrics
test_error = np.abs(test_y - prediction)
mean_error = np.mean(test_error)
min_error = np.min(test_error)
max_error = np.max(test_error)
std_error = np.std(test_error)

print("Mean Error:" + str(mean_error))
print("Min Error:" + str(min_error))
print("Max Error:" + str(max_error))
print("Std Error:" + str(std_error))

plot_loss(history)

plt.yscale('log')
plt.scatter(test_y, test_error)
plt.xlabel("True Values")
plt.ylabel("Error")
plt.show()

plt.scatter(test_y, prediction)
plt.xlabel("index")
plt.ylabel("Prediction")
plt.show()

