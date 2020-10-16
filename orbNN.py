import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
from keras.backend import sigmoid
import keras.backend as K
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Activation, BatchNormalization, Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.utils.generic_utils import get_custom_objects
from keras.optimizers import SGD

from keras.regularizers import l1
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from keras.utils import to_categorical
from scipy.stats import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

epochs = 25
batch_size = 25

gpus = tf.config.experimental.list_physical_devices('GPU')


# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(gpus[0], [
#             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#     except RuntimeError as e:
#         print(e)


def swish(x, beta=1):
    return (x * sigmoid(beta * x))


get_custom_objects().update({'swish': Activation(swish)})


def tilted_loss(q, y, f):
    e = (y - f)
    return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)


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
    start_epoch = 0
    fig = plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'][start_epoch:])
    plt.plot(history.history['val_loss'][start_epoch:])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


def plot_accuracy(history):
    start_epoch = 0
    fig = plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'][start_epoch:])
    plt.plot(history.history['val_accuracy'][start_epoch:])
    plt.title('model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


opt = SGD(lr=0.01)

data_y = np.load("data_y.npy")
data_y = pd.qcut(data_y, 2).codes
# data_y = to_categorical(data_y)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(data_y)
data_y = encoder.transform(data_y)


# Load up the clean data
clean_data = np.load('cleanData.npy', allow_pickle=True)
data = clean_data

print("Loading Data Set")
print("Loading orb feature data")

features_orb_0 = np.load("features_orb_0.npy", allow_pickle=True)
features_orb_1 = np.load("features_orb_1.npy", allow_pickle=True)
features_orb_2 = np.load("features_orb_2.npy", allow_pickle=True)
features_orb_3 = np.load("features_orb_3.npy", allow_pickle=True)
features_orb_4 = np.load("features_orb_4.npy", allow_pickle=True)
features_orb_5 = np.load("features_orb_5.npy", allow_pickle=True)
features_orb_6 = np.load("features_orb_6.npy", allow_pickle=True)
features_orb_7 = np.load("features_orb_7.npy", allow_pickle=True)
features_orb_8 = np.load("features_orb_8.npy", allow_pickle=True)
features_orb_9 = np.load("features_orb_9.npy", allow_pickle=True)
features = np.concatenate((features_orb_0,
                           features_orb_1,
                           features_orb_2,
                           features_orb_3,
                           features_orb_4,
                           features_orb_5,
                           features_orb_6,
                           features_orb_7,
                           features_orb_8,
                           features_orb_9), axis=0)

# df['quantile'] = pd.qcut(data_y['b'], 2, labels=False)

# Remove outliers
# z_threshold = 1
# print("Removing Outliers")
# z = np.abs(stats.zscore(data[:, 6].astype(int)))
#
# print(data.shape)
# print(data_y.shape)
#
# data = data[(z < z_threshold)]
# data_y = data_y[(z < z_threshold)]
#
# print("Outliers Removed")
# print(data.shape)
# print(data_y.shape)

# Train Test validation split
np.random.rand(42)

indices = np.random.permutation(data.shape[0])
index = int(0.8 * len(indices))
percent = int(0.1 * len(indices))

training_idx, test_idx, val_idx = indices[:index], indices[index:index + percent], indices[index + percent:]
train_data, test_data, val_data = data[training_idx, :], data[test_idx, :], data[val_idx, :]

# training_idx, test_idx = indices[:index], indices[index:index + percent]
# train_data, test_data = data[training_idx, :], data[test_idx, :]

print(features.shape)
# train, test = features[training_idx], features[test_idx]
# train_y, test_y = data_y[training_idx], data_y[test_idx]

train, test, val = features[training_idx], features[test_idx], features[val_idx]
train_y, test_y, val_y = data_y[training_idx], data_y[test_idx], data_y[val_idx]

# yPlot = data_y
# y_pos = np.sort(yPlot.astype(int))
# sns.set(color_codes=True)
# sns.distplot(y_pos, bins=20)
# plt.show()
#
# yPlot = data_y
# y_pos = np.sort(yPlot)
# sns.set(color_codes=True)
# sns.distplot(y_pos)
# plt.show()

# Creating a checkpointer
checkpointer = ModelCheckpoint(filepath='scratchmodel.best.hdf5', monitor='val_accuracy',
                               verbose=1, save_best_only=True)

# EarlyStoppingByLossVal(monitor='val_loss', value=0.001, verbose=1),
callbacks = [checkpointer]


def baseline_model():
    model_scratch = Sequential()
    model_scratch.add(Dense(len(train), activation=swish, input_shape=train.shape[1:], activity_regularizer=l1(0.001)))
    model_scratch.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model_scratch.add(Dense(128, activation=swish, activity_regularizer=l1(0.001)))
    model_scratch.add(Dropout(0.2))
    model_scratch.add(Dense(1, activation='sigmoid'))
    # model_scratch.compile(loss=lambda y, f: tilted_loss(quantile, y, f),
    #                       optimizer='adagrad', metrics=['mse', 'mae', 'mape'])

    model_scratch.compile(loss='binary_crossentropy',
                          optimizer=opt, metrics=['accuracy'])
    return model_scratch


quantile = 0.9

# # Reshape data for NN
# # train = train.reshape(len(train), 1000, 32, 1)
# # test = test.reshape(len(test), 1000, 32, 1)
# # val = val.reshape(len(val), 1000, 32, 1)

# Reshape data for StandardScalar
train = train.reshape(len(train), 1000 * 32)
test = test.reshape(len(test), 1000 * 32)
val = val.reshape(len(val), 1000 * 32)

estimators = []
# estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=100, verbose=False)
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=1)))

pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, train, train_y, cv=kfold)
print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# history = baseline_model().fit(train, train_y, batch_size=batch_size, epochs=epochs,
#                                validation_data=(val, val_y), callbacks=callbacks,
#                                verbose=1, shuffle=True)
# test_loss, test_acc = estimator.evaluate(test,  test_y, verbose=2)
# print('\nTest accuracy:', test_acc)

# predictions = baseline_model().predict(test)

# test_loss, test_acc = baseline_model().evaluate(test, test_y, verbose=2)
#
# print('\nTest accuracy:', test_acc)

# np.argmax(predictions[0])

# def plot_value_array(i, predictions_array, true_label):
#     true_label = true_label[i]
#     plt.grid(False)
#     plt.xticks(range(4))
#     plt.yticks([])
#     thisplot = plt.bar(range(4), predictions_array, color="#777777")
#     plt.ylim([0, 1])
#     predicted_label = np.argmax(predictions_array)
#
#     thisplot[predicted_label].set_color('red')
#     thisplot[np.argmax(true_label)].set_color('blue')
#     plt.show()
#
#
# for i in range(len(test)):
#     plot_value_array(i, predictions[i], test_y)


# print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

# history = baseline_model().fit(train, train_y, batch_size=batch_size, epochs=epochs,
#                                validation_data=(val, val_y),
#                                verbose=1, shuffle=True)

# prediction = baseline_model().predict(test)

# test_error = np.abs(test_y - prediction)
# mean_error = np.mean(test_error)
# min_error = np.min(test_error)
# max_error = np.max(test_error)
# std_error = np.std(test_error)
#
# print("Mean Error:" + str(mean_error))
# print("Min Error:" + str(min_error))
# print("Max Error:" + str(max_error))
# print("Std Error:" + str(std_error))


# plot_loss(history)
#
# plt.yscale('log')
# plt.scatter(test_y, test_error)
# plt.xlabel("True Values")
# plt.ylabel("Error")
# plt.show()

# plt.scatter(test_y, predictions)
# plt.xlabel("index")
# plt.ylabel("Prediction")
# plt.show()
#
# plot_accuracy(history)
# plot_loss(history)
# print(predictions)
# print(test_y)
# real_accuracy = 0
# for i in range(len(predictions) - 1):
#     if np.argmax(predictions[i]) == test_y[i]:
#         real_accuracy = real_accuracy + 1
# # print('%s => %d (expected %d)' % (i, np.argmax(predictions[i]), test_y[i]))
#
# print('\nActual accuracy:', real_accuracy / np.size(predictions))
