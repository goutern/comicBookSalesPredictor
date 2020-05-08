import warnings

import cv2
import cv2.cuda as cv2cuda
import matplotlib.pyplot as plt
import numpy as np
import pylab
import seaborn as sns
# Importing Keras libraries
import tensorflow as tf
from keras import Input
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.backend import sigmoid
import keras.backend as K
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Activation, BatchNormalization, MaxPooling1D, GlobalAveragePooling1D, Conv2D, \
    GlobalAveragePooling2D, Conv1D, MaxPool1D
from keras.layers import Dense, MaxPooling2D
from keras.layers import Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from keras.utils.generic_utils import get_custom_objects
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.stats import stats
# Train whole data then test on decades
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


# Importing sklearn libraries

epochs = 100
batch_size = 32
image_count = 10
limit_data = False
# max_samples = 100
# test_run = True
generate_data = False
load_image_data = True
load_features = False
load_features_flat = False
show_dist_graphs = False
normalize = False
use_orb = False

limit_memory = False

remove_outliers = True
threshold = 1


def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

get_custom_objects().update({'swish': Activation(swish)})


def tilted_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

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
    start_epoch = 2
    fig = plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'][start_epoch:])
    plt.plot(history.history['val_loss'][start_epoch:])
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

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb_feature_count = 1000

orb = cv2.ORB_create(nfeatures=orb_feature_count, scoreType=cv2.ORB_FAST_SCORE)

def get_orb_features(dataSource):
    x_scratch = np.zeros(shape=(1,orb_feature_count, 32))
    count = 0
    for comic in dataSource:
        if(count % 100 == 0):
            print("Comic:" + str(count) + " of " + str(len(dataSource)))
        count = count + 1
        img_path = comic[3]
        # print(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # descriptors = sift.detectAndCompute(img, None)
        # descriptors = surf.detectAndCompute(img, None)
        keypoints, descriptors = orb.detectAndCompute(img, None)
        try:
            while len(descriptors) < orb_feature_count:
                newrow = np.zeros(shape=(32))
                descriptors = np.vstack((descriptors,newrow))
            while len(descriptors) > orb_feature_count:
                descriptors = np.delete(descriptors, 0,0)
            descriptors = descriptors[np.newaxis,:,:]
        except (RuntimeError, TypeError, NameError):
            descriptors = np.zeros(shape=(1,orb_feature_count, 32))
        x_scratch = np.vstack((x_scratch,descriptors))
    x = np.asarray(x_scratch)
    x = np.delete(x, 0,0)

    # features = pre_model.predict(x, batch_size=batch_size)
    # features_flatten = features.reshape((features.shape[0], 7 * 7 * 512))
    return x




# img = cv2.drawKeypoints(img, keypoints_orb, None)
# cv2.imshow("Image", img)
# cv2.waitKey(0)

# img_data = image.img_to_array(img)
#         # img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
# img_data = np.expand_dims(img_data, axis=0)
# img_data = np.squeeze(img_data, axis=0)

if limit_data:
    clean_data = clean_data[:image_count]
data_y = []
data = clean_data

def create_y(data):
    for comic in data:
        data_y.append(int(comic[6]))
    np.save("data_y.npy", data_y)
    np.savetxt('data_y.csv', data_y, delimiter=',')

if generate_data:
    if (use_orb):
        print("getting Orb")
        features = get_orb_features(data)
        np.save("features_orb.npy", features)
        print("Orb Saved")
    else:
        create_y(data)
        model = VGG16( weights="imagenet",include_top=False)
        # model = VGG16( include_top=False)
        model.summary()

        print("Creating Features")
        image_data, features, features_flatten = create_features(clean_data, model)

        np.save("image_data.npy", image_data)
        np.save("features.npy", features)
        np.save("features_flatten.npy", features_flatten)


print("Loading Data Set")
if load_image_data:
    print("Loading image data")
    image_data = np.load("image_data.npy")

if load_features:
    print("Loading feature data")
    if use_orb:
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
    else:
        features = np.load("features.npy")
if load_features_flat:
    print("Loading flattened feature data")
    features_flatten = np.load("features_flatten.npy")
data_y = np.load("data_y.npy")
if limit_data:
    data_y = data_y[:image_count]



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
np.random.rand(42)
if limit_data:
    print("Test Run Activated Running on Reduced Data Set")
    print("Image Count: " + str(image_count))
    indices = np.random.permutation(data.shape[0])[:image_count]
else:
    indices = np.random.permutation(data.shape[0])
index = int(0.8 * len(indices))
percent = int(0.1 * len(indices))


#train test validation split
training_idx, test_idx, val_idx = indices[:index], indices[index:index+percent], indices[index+percent:]
train_data, test_data, val_data = data[training_idx,:], data[test_idx,:], data[val_idx,:]


# features = np.reshape(features, (features.shape[0],112,112,32))
# # feature = np.reshape(feature, (224,112))
# pic = features[1]
# pylab.imshow(pic)
# pylab.show()
# features = np.squeeze(features, axis=0)


if load_image_data:
    train, test, val = image_data[training_idx,:], image_data[test_idx,:], image_data[val_idx,:]

if load_features:
    print(features.shape)
    train, test, val = features[training_idx], features[test_idx], features[val_idx]

if load_features_flat:
    train, test, val = features_flatten[training_idx,:], features_flatten[test_idx,:], features_flatten[val_idx,:]


train_y, test_y, val_y = data_y[training_idx], data_y[test_idx], data_y[val_idx]
if normalize:
    exp = .2
    train_y = np.power(train_y.astype(float),exp)
    val_y = np.power(val_y.astype(float),exp)
    scaler = MinMaxScaler()

    # train_y = train_y.reshape(-1, 1)
    # # train_y = scaler.fit(train_y)
    # normalized_train_y = scaler.fit_transform(train_y)
    # inverse_train_y = scaler.inverse_transform(normalized_train_y)
    #
    # val_y = val_y.reshape(-1, 1)
    # # val_y = scaler.fit(val_y)
    # normalized_val_y = scaler.fit_transform(val_y)
    # inverse_train_y = scaler.inverse_transform(normalized_val_y)

    # train_y = np.power(train_y.astype(float), exp)
    # val_y = np.power(val_y.astype(float),exp)





if show_dist_graphs:
    yPlot = data_y
    y_pos = np.sort(yPlot.astype(int))
    sns.set(color_codes=True)
    sns.distplot(y_pos,bins=20)
    plt.show()

    if normalize:
        yPlot = data_y
        exp = .2
        yPlot = np.power(yPlot.astype(float),exp)
        yPlot = yPlot.reshape(-1, 1)
        # yPlot = scaler.fit(yPlot)
        # normalized_yPlot = scaler.fit_transform(yPlot)
        # inverse_train_y = scaler.inverse_transform(normalized_yPlot)

        y_pos = np.sort(yPlot)
        sns.set(color_codes=True)
        sns.distplot(y_pos)
        plt.show()

    # yPlot = train_y
    # y_pos = np.sort(yPlot.astype(int))
    # x_pos = np.arange(len(yPlot))
    # plt.scatter(x_pos, y_pos, alpha=0.5, color='blue')
    # # plt.show()
    #
    # yPlot = test_y
    # y_pos = np.sort(yPlot.astype(int))
    # x_pos = np.arange(len(yPlot))
    # plt.scatter(x_pos, y_pos, alpha=0.5, color='red')
    # # plt.show()
    #
    # yPlot = val_y
    # y_pos = np.sort(yPlot.astype(int))
    # x_pos = np.arange(len(yPlot))
    # plt.scatter(x_pos, y_pos, alpha=0.5,color='green')
    # plt.show()



# Creating a checkpointer
checkpointer = ModelCheckpoint(filepath='scratchmodel.best.hdf5',
                               verbose=1, save_best_only=True)

callbacks = [
    EarlyStoppingByLossVal(monitor='val_loss', value=0.001, verbose=1),
    checkpointer]


optimizer = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
# optimizer = SGD(lr=0.01, momentum=0.9)
def baseline_model():
    # model = Sequential()
    # model.add(MaxPooling1D())

    # model = Sequential()
    # model.add(Conv1D(3,3, activation='relu', input_shape=train.shape[1:]))
    # model.add(Conv1D(3,3, activation='relu'))
    # model.add(MaxPool1D(2))
    # model.add(Dropout(0.5))
    #
    # model.add(Conv1D(3,3, activation='relu'))
    # model.add(Conv1D(3,3, activation='relu'))
    # model.add(MaxPool1D(2))
    # model.add(Dropout(0.5))
    # model.add(Flatten())

    # model.add(Dense(1024, activation="swish", input_shape=train.shape[1:]))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation=swish))
    #
    # model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'mae','mape'])



    # model.add(Dense(1024, activation='relu', input_shape=train.shape[1:]))
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dense(1, activation='relu'))

    # model = Sequential()
    # model.add(GlobalAveragePooling1D(input_shape=train.shape[1:]))
    # model.add(Flatten())
    # model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))

    #     # model.add(Dense(256, activation=swish))
    #     # model.add(Dropout(0.5))
    #     # # model.add(Dense(1024, activation="relu"))
    #     # # model.add(Dropout(0.5))
    #     # # model.add(Dense(64, activation="relu"))
    #     # # model.add(Dropout(0.5))
    #     # # model.add(Dense(1, activation="relu"))
    #     # model.add(Dense(1))
    # model.add(Dense(1024, activation=swish,input_shape=train.shape[1:]))
    # model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    # model.add(Dropout(0.5))
    # model.add(Dense(1024, activation=swish, use_bias=True))
    # model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    # model.add(Dropout(0.5))
    # model.add(Dense(256, activation=swish, use_bias=True))
    # model.add(Flatten())
    # model.add(Dense(1))
    # model.compile(loss='mse', optimizer="adagrad", metrics=['mse', 'mae','mape'])
    #
    # Building up a Sequential model
    stride_val = 2

    model_scratch = Sequential()
    model_scratch.add(Conv2D(32, (3, 3), activation=swish, strides=stride_val, input_shape=train.shape[1:]))

    if stride_val == 1: model_scratch.add(MaxPooling2D(pool_size=(2, 2)))

    model_scratch.add(Conv2D(64, (3, 3), activation=swish, strides=stride_val))
    if stride_val == 1:model_scratch.add(MaxPooling2D(pool_size=(2, 2)))

    model_scratch.add(Conv2D(64, (3, 3), activation=swish, strides=stride_val))
    if stride_val == 1:model_scratch.add(MaxPooling2D(pool_size=(2, 2)))

    model_scratch.add(Conv2D(128, (3, 3), activation=swish, strides=stride_val))
    if stride_val == 1:model_scratch.add(MaxPooling2D(pool_size=(2, 2)))

    model_scratch.add(Conv2D(256, (3, 3), activation=swish, strides=stride_val))
    if stride_val == 1:model_scratch.add(MaxPooling2D(pool_size=(2, 2)))

    model_scratch.add(Conv2D(512, (3, 3), activation=swish, strides=stride_val))
    if stride_val == 1:model_scratch.add(MaxPooling2D(pool_size=(2, 2)))


    model_scratch.add(Flatten())
    model_scratch.add(Dense(512, activation=swish))
    model_scratch.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model_scratch.add(Dropout(0.5))
    model_scratch.add(Dense(512, activation=swish))
    model_scratch.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model_scratch.add(Dropout(0.5))
    model_scratch.add(Dense(1, activation=swish))
    model_scratch.compile(loss=lambda y, f: tilted_loss(quantile, y, f), optimizer='adagrad' , metrics=['mse', 'mae','mape'])
    model_scratch.summary()
    return model_scratch

quantile = 0.9
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size= 100, verbose=False)

# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, train_features, y_train, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# train_flat_trick = np.reshape(train, len(train)^2, 1)

# train = train.reshape(len(train),1000,32,1)
# test = test.reshape(len(test),1000,32,1)
# val = val.reshape(len(val),1000,32,1)

# ridge=Ridge()
# # parameters= {'alpha':[x for x in [.001,.0015,0.002]]}
#
# nsamples, nx, ny = train.shape
# d2_train_dataset = train.reshape((nsamples,nx*ny))
#
# nsamples, nx, ny = test.shape
# d2_test_dataset = test.reshape((nsamples,nx*ny))
#
# nsamples, nx, ny = val.shape
# d2_val_dataset = val.reshape((nsamples,nx*ny))

# ridge_reg=GridSearchCV(ridge, param_grid=parameters)
# ridge_reg.fit(d2_train_dataset,train_y)
# print("The best value of Alpha is: ",ridge_reg.best_params_)


# ridge_mod=Ridge(alpha=.0015)
# ridge_mod.fit(d2_train_dataset,train_y)
# y_pred_train=ridge_mod.predict(d2_train_dataset)
# y_pred_test=ridge_mod.predict(d2_test_dataset)
# y_pred_val=ridge_mod.predict(d2_val_dataset)
#
# print('Root Mean Square Error train = ' + str(np.sqrt(mean_squared_error(train_y, y_pred_train))))
# print('Root Mean Square Error val = ' + str(np.sqrt(mean_squared_error(val_y, y_pred_val))))
# print('Root Mean Square Error test = ' + str(np.sqrt(mean_squared_error(test_y, y_pred_test))))

history = estimator.fit(train, train_y, batch_size=batch_size, epochs=epochs,
          validation_data=(val, val_y), callbacks = callbacks,
          verbose=1, shuffle=True)

prediction = estimator.predict(test)
# pic = features[0,:,:,128]
# pylab.imshow(pic)
# pylab.show()
if normalize:
    prediction = np.power(prediction, 5)
    # prediction = prediction.reshape(-1,1)
    # prediction = scaler.inverse_transform(prediction)
test_error = np.abs(test_y - prediction)
mean_error = np.mean(test_error)
min_error = np.min(test_error)
max_error = np.max(test_error)
std_error = np.std(test_error)



print("Mean Error:" + str(mean_error))
print("Min Error:" + str(min_error))
print("Max Error:" + str(max_error))
print("Std Error:" + str(std_error))

# print('Root Mean Square Error train = ' + str(np.sqrt(mean_squared_error(train_y, y_pred_train))))
# print('Root Mean Square Error val = ' + str(np.sqrt(mean_squared_error(val_y, y_pred_val))))
# print('Root Mean Square Error test = ' + str(np.sqrt(mean_squared_error(test_y, y_pred_test))))


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


# y_pos = np.sort(test_error.astype(int))
# x_pos = np.arange(len(y_pos))
# plt.scatter(x_pos, y_pos, alpha=0.5, color='red')
# plt.show()
#
# y_pos = np.sort(prediction.astype(int))
# x_pos = np.arange(len(y_pos))
# plt.scatter(x_pos, y_pos, alpha=0.5, color='yellow')
# plt.show()

