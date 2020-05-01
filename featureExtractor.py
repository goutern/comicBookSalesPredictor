import os
import warnings

from keras.backend import sigmoid
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
import cv2
import numpy as np
import pyspark
import matplotlib.pyplot as plt
import seaborn as sns
from keras.wrappers.scikit_learn import KerasRegressor
from pyspark.ml.feature import VectorAssembler
from scipy.stats import stats
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Importing sklearn libraries
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing Keras libraries
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from keras.utils import np_utils
from keras.models import Sequential
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint, Callback
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Conv2D, MaxPooling2D, LSTM, GlobalMaxPooling2D, SpatialDropout1D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D

from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation

from hypopt import GridSearch
from sklearn.neural_network import MLPClassifier



epochs = 25
batch_size = 32
image_count = 100
test_run = False
generate_data = False
load_image_data = False
load_features = True
load_features_flat = False
show_dist_graphs = True

limit_memory = False

remove_outliers = True
threshold = 1


def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish': Activation(swish)})


def create_features(dataSource, pre_model):
    x_scratch = []
    for comic in dataSource:
        img_path = comic[3]
        # print(img_path)
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        # img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = imagenet_utils.preprocess_input(img_data)
        x_scratch.append(img_data)


    x = np.vstack(x_scratch)
    features = pre_model.predict(x, batch_size=batch_size)
    features_flatten = features.reshape((features.shape[0], 7 * 7 * 512))
    return x, features, features_flatten
    # return x, features



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
    fig = plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

if limit_memory:
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

def create_y(data):
    for comic in data:
        data_y.append(int(comic[6]))
    np.save("data_y.npy", data_y)


if generate_data:
    create_y(data)
    model = VGG16(weights='imagenet', include_top=False)
    model.summary()

    print("Creating Features")
    image_data, features, features_flatten = create_features(clean_data, model)

    np.save("image_data.npy", image_data)
    np.save("features.npy", features)
    np.save("features_flatten.npy", features_flatten)

else:
    print("Loading Data Set")
    if load_image_data:
        print("Loading image data")
        image_data = np.load("image_data.npy")
    if load_features:
        print("Loading feature data")
        features = np.load("features.npy")
    if load_features_flat:
        print("Loading flattend feature data")
        features_flatten = np.load("features_flatten.npy")
    data_y = np.load("data_y.npy")



if remove_outliers:
    #Remove outliers
    print("Removing Outliers")
    z = np.abs(stats.zscore(data[:,6].astype(int)))

    # print(np.where(z > threshold))
    print(data.shape)
    print(data_y.shape)

    data = data[(z < threshold)]
    data_y = data_y[(z < threshold)]

    print("Outliers Removed")
    print(data.shape)
    print(data_y.shape)


#used to reduce the image pool to run faster tests
if test_run:
    print("Test Run Activated Running on Reduced Data Set")
    print("Image Count: " + str(image_count))
    indices = np.random.permutation(data.shape[0])[:image_count]
else:
    indices = np.random.permutation(data.shape[0])
index = int(0.6 * len(indices))
percent = int(0.2 * len(indices))


#train test validation split
training_idx, test_idx, val_idx = indices[:index], indices[index:index+percent], indices[index+percent:]
train, test, val = data[training_idx,:], data[test_idx,:], data[val_idx,:]

if load_image_data:
    train_image, test_image, val_image = image_data[training_idx,:], image_data[test_idx,:], image_data[val_idx,:]

if load_features:
    train_features, test_features, val_features = features[training_idx,:], features[test_idx,:], features[val_idx,:]

if load_features_flat:
    train_features_flatten, test_features_flatten, val_features_flatten = features_flatten[training_idx,:], features_flatten[test_idx,:], features_flatten[val_idx,:]

train_y, test_y, val_y = data_y[training_idx], data_y[test_idx], data_y[val_idx]




if show_dist_graphs:
    yPlot = data_y
    y_pos = np.sort(yPlot.astype(int))
    sns.set(color_codes=True)
    sns.distplot(y_pos)
    plt.show()

    yPlot = train_y
    y_pos = np.sort(yPlot.astype(int))
    x_pos = np.arange(len(yPlot))
    plt.scatter(x_pos, y_pos, alpha=0.5, color='red')
    plt.show()

    yPlot = test_y
    y_pos = np.sort(yPlot.astype(int))
    x_pos = np.arange(len(yPlot))
    plt.scatter(x_pos, y_pos, alpha=0.5, color='blue')
    plt.show()

    yPlot = val_y
    y_pos = np.sort(yPlot.astype(int))
    x_pos = np.arange(len(yPlot))
    plt.scatter(x_pos, y_pos, alpha=0.5,color='green')
    plt.show()



# Creating a checkpointer
checkpointer = ModelCheckpoint(filepath='scratchmodel.best.hdf5',
                               verbose=1, save_best_only=True)

callbacks = [
    EarlyStoppingByLossVal(monitor='val_loss', value=0.00001, verbose=1),
    checkpointer]


def baseline_model():
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=train_features.shape[1:]))
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(512, activation=swish))
    model.add(Dropout(0.1))
    # model.add(Dense(64, activation=swish))
    # model.add(Dropout(0.1))
    model.add(Dense(1, activation=swish))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=100, verbose=False)

# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, train_features, y_train, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

history = estimator.fit(train_features, train_y, batch_size=batch_size, epochs=epochs,
          validation_data=(val_features, val_y), callbacks=callbacks,
          verbose=1, shuffle=True)
prediction = estimator.predict(test_features)

test_error =  np.abs(test_y - prediction)
mean_error = np.mean(test_error)
min_error = np.min(test_error)
max_error = np.max(test_error)
std_error = np.std(test_error)

plot_loss(history)

plt.scatter(test_y, prediction)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()


y_pos = np.sort(test_error.astype(int))
x_pos = np.arange(len(y_pos))
plt.scatter(x_pos, y_pos, alpha=0.5, color='red')
plt.show()

y_pos = np.sort(prediction.astype(int))
x_pos = np.arange(len(y_pos))
plt.scatter(x_pos, y_pos, alpha=0.5, color='yellow')
plt.show()

print("Mean Error:" + str(mean_error))
print("Min Error:" + str(min_error))
print("Max Error:" + str(max_error))
print("Std Error:" + str(std_error))
