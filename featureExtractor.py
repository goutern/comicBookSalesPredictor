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


def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish': Activation(swish)})


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

#get out "trial development" data

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024
#                                                                                                                         )])
#   except RuntimeError as e:
#     print(e)

data = np.load('cleanData.npy',allow_pickle=True)
# sortedData = data[np.argsort(data[:,6])]
epochs = 25
batch_size = 128
image_count = 100
test_run = False
load_train_data = True
show_dist_graphs = False
threshold = 3
seed = 1
# data = data[:10000,:]
# x is your dataset
# x = np.random.rand(100, 5)
# indices = np.random.permutation(data.shape[0])[:image_count]

#
#
# print(y_train[0:10])
#
#
# print("Training data available in 3000 classes")
# print([train_y.count(i) for i in range(0, 11)])


# # Show our data distibution
# y_pos = np.arange(num_classes)
# counts = [train_y.count(i) for i in range(0, num_classes)]
#
# plt.barh(y_pos, counts, align='center', alpha=0.5)
# # plt.yticks(y_pos)
# plt.xlabel('Counts')
# plt.title('Train Data Class Distribution')
# plt.show()

# Show our data distibution

z = np.abs(stats.zscore(data[:,6].astype(int)))

print(np.where(z > threshold))
print(data.shape)

data = data[(z < threshold)]
print(data.shape)


# Q1 = data[:,6].quantile(0.25)
# Q3 = data[:,6].quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)
#
# data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
# print(data.shape)

if test_run:
    indices = np.random.permutation(data.shape[0])[:image_count]
else:
    indices = np.random.permutation(data.shape[0])
index = int(0.6 * len(indices))
percent = int(0.2 * len(indices))

training_idx, test_idx, val_idx = indices[:index], indices[index:index+percent], indices[index+percent:]
training, test, val = data[training_idx,:], data[test_idx,:], data[val_idx,:]

model = VGG16(weights='imagenet', include_top=False)
model.summary()
# data = np.load('cleanData2019L.npy',allow_pickle=True)
# data = np.delete(data, 0,0)
# valData = np.load('cleanData2018VAL.npy', allow_pickle=True)
# valData = np.delete(data, 0,0);
# testData = np.load('cleanData2017test.npy', allow_pickle=True)
# testData =  np.delete(data, 0,0)
# featureData = np.zeros(shape=(1,10))
train_y = []
val_y = []
test_y = []
# num_classes = 1150

#get the sales numbers
for comic in training:
    train_y.append(int(comic[6]))

for comic in val:
    val_y.append(int(comic[6]))

for comic in test:
    test_y.append(int(comic[6]))

# print(train_y[0:10])

#incode our sales numbers categories using one hot encoding
# y_train = np_utils.to_categorical(train_y, num_classes)
# y_val = np_utils.to_categorical(val_y, num_classes)
# y_test = np_utils.to_categorical(test_y, num_classes)
y_train = train_y
y_val = val_y
y_test = test_y




if show_dist_graphs:
    yPlot = data[:,6]
    y_pos = np.sort(yPlot.astype(int))
    sns.set(color_codes=True)
    sns.distplot(y_pos)
    plt.show()

# sns.boxplot(y_pos)
# plt.show()


    yPlot = training[:,6]
    y_pos = np.sort(yPlot.astype(int))
    x_pos = np.arange(len(yPlot))

    plt.scatter(x_pos, y_pos, alpha=0.5, color='red')
    plt.show()



    yPlot = test[:,6]
    y_pos = np.sort(yPlot.astype(int))
    x_pos = np.arange(len(yPlot))
    plt.scatter(x_pos, y_pos, alpha=0.5, color='blue')
    plt.show()

    yPlot = val[:,6]
    y_pos = np.sort(yPlot.astype(int))
    x_pos = np.arange(len(yPlot))
    plt.scatter(x_pos, y_pos, alpha=0.5,color='green')
    plt.show()


# counts = [data[:,:6]]
# print(1)
# plt.barh(y_pos,x_pos, align='center', alpha=0.5)
# print(1)
# plt.yticks(y_pos)
# print(1)
# plt.xlabel('Percent Total')
# print(1)
# plt.title('Train Data Class Distribution')
# print(1)
# plt.show()
# print(1)



# # Pie chart, where the slices will be ordered and plotted counter-clockwise:
# labels = np.arange(num_classes)
# sizes = [train_y.count(i) for i in range(0, num_classes)]
# # explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
#
# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
# plt.show()


# Extract the features using the pretrained VGG
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



# train_x, train_features, train_features_flatten = create_features(data, model)
# print(train_x.shape, train_features.shape, train_features_flatten.shape)
# val_x, val_features, val_features_flatten = create_features(valData, model)
# test_x, test_features, test_features_flatten = create_features(testData, model)

if load_train_data:
    print("Loading Features")
    train_features = np.load("train_features.npy")
    val_features = np.load("val_features.npy")
    test_features = np.load("test_features.npy")
else:
    print("Creating Features")
    train_x, train_features, train_features_flatten = create_features(training, model)
    print(train_x.shape, train_features.shape)
    val_x, val_features,val_features_flatten = create_features(val, model)
    print(val_x.shape, val_features.shape)
    test_x, test_features,test_features_flatten = create_features(test, model)
    print(test_x.shape, test_features.shape)

    np.save("train_features.npy", train_features)
    np.save("val_features.npy", val_features)
    np.save("test_features.npy", test_features)



# nsamples, nx, ny = train_x.shape[:-1]
# d2_train_dataset = train_x.reshape((nsamples,nx*ny))
# regressor = LinearRegression()
# regressor.fit(d2_train_dataset, y_train) #training the algorithm
# val_x, val_features, val_features_flatten = create_features(val, model)
# test_x, test_features, test_features_flatten = create_features(test, model)

# Creating a checkpointer
checkpointer = ModelCheckpoint(filepath='scratchmodel.best.hdf5',
                               verbose=1, save_best_only=True)
#
# # Building up a Sequential model
# model_scratch = Sequential()
# model_scratch.add(Conv2D(32, (3, 3), activation='relu', input_shape=train_x.shape[1:]))
# model_scratch.add(MaxPooling2D(pool_size=(2, 2)))
#
# model_scratch.add(Conv2D(64, (3, 3), activation='relu'))
# model_scratch.add(MaxPooling2D(pool_size=(2, 2)))
#
# model_scratch.add(Conv2D(64, (3, 3), activation='relu'))
# model_scratch.add(MaxPooling2D(pool_size=(2, 2)))
#
# model_scratch.add(Conv2D(128, (3, 3), activation='relu'))
# model_scratch.add(MaxPooling2D(pool_size=(2, 2)))
#
#
#
# model_scratch.add(GlobalAveragePooling2D())
# model_scratch.add(Dense(64, activation='relu'))
# # model_scratch.add(Dropout(0.1))
# model_scratch.add(Dense(64, activation='relu'))
# model_scratch.add(Dropout(0.1))
# model_scratch.add(Dense(1, activation='relu'))
# model_scratch.summary()
#
# model_scratch.compile(loss='mean_squared_error', optimizer='adam',
#                       metrics=['accuracy'])
callbacks = [
    EarlyStoppingByLossVal(monitor='val_loss', value=0.00001, verbose=1),
    # EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    checkpointer]
# Fitting the model on the train data and labels.
# history = model_scratch.fit(train_x, y_train,
#                             batch_size=32, epochs=epochs,
#                             verbose=1, callbacks=callbacks,
#                             validation_data=(val_x, y_val), shuffle=True)

# model = Sequential()
# model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
# model.add(Dense(1, kernel_initializer='normal'))


def plot_loss(history):
    fig = plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')

    # plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


# plot_acc_loss(history)
#
# preds = np.argmax(model_scratch.predict(test_x), axis=1)
# print("\nAccuracy on Test Data: ", accuracy_score(test_y, preds))
# print("\nNumber of correctly identified imgaes: ",
#       accuracy_score(test_y, preds, normalize=False),"\n")
# # confusion_matrix(test_y, preds, labels=range(0,num_classes))



# model = MLPClassifier()
#
#
# model.fit(train_features, y_train)
#
#
#
#
#
# model_transfer = Sequential()
# model_transfer.add(GlobalAveragePooling2D(input_shape=train_features.shape[1:]))
# model_transfer.add(Dense(64, activation='relu'))
# # model_scratch.add(Dropout(0.1))
# model_transfer.add(Dense(64, activation='relu'))
# # model_transfer.add(Dropout(0.1))
# model_transfer.add(Dense(1, activation='relu'))
# model_transfer.summary()
#
# model_transfer.compile(loss='mean_squared_error', optimizer='adam',
#               metrics=['accuracy'])
# history = model_transfer.fit(train_features, y_train, batch_size=32, epochs=epochs,
#           validation_data=(val_features, y_val), callbacks=callbacks,
#           verbose=1, shuffle=True)

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

history = estimator.fit(train_features, y_train, batch_size=batch_size, epochs=epochs,
          validation_data=(val_features, y_val), callbacks=callbacks,
          verbose=1, shuffle=True)
prediction = estimator.predict(train_features)

train_error =  np.abs(y_train - prediction)
mean_error = np.mean(train_error)
min_error = np.min(train_error)
max_error = np.max(train_error)
std_error = np.std(train_error)

plot_loss(history)


y_pos = np.sort(train_error.astype(int))
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
# preds = np.argmax(prediction.predict(test_features_flatten), axis=1)
# print("\nAccuracy on Test Data: ", accuracy_score(test_y, prediction))
# print("\nNumber of correctly identified imgaes: ",
#       accuracy_score(test_y, prediction, normalize=False),"\n")
# confusion_matrix(test_y, preds, labels=range(0,num_classes))

#
# print("1")
# param_grid = [{'C': [0.1, 1, 10], 'solver': ['newton-cg', 'lbfgs']}]
# print("2")
# # wait()
# # # Grid-search all parameter combinations using a validation set.
# opt = GridSearch(model=LogisticRegression(class_weight='balanced', multi_class="auto",
#                                           max_iter=4000, random_state=1), param_grid=param_grid, num_threads=4, parallelize=False)
# print("3")
# opt.fit(train_features_flatten, train_y, val_features_flatten, val_y, scoring='accuracy')
# print("4")
# print(opt.get_best_params())
# print("5")
# opt.score(test_features_flatten, test_y)
# print("6")
# preds = opt.predict(test_features_flatten)
# print("7")
# print("\nAccuracy on Test Data: ", accuracy_score(test_y, preds))
# print("8")
# print("\nNumber of correctly identified imgaes: ",
# accuracy_score(test_y, preds, normalize=False),"\n")
# confusion_matrix(test_y, preds, labels=range(0,num_classes))