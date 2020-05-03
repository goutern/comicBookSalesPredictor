import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pylab
import seaborn as sns
# Importing Keras libraries
import tensorflow as tf
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.backend import sigmoid
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Activation, BatchNormalization, MaxPooling1D
from keras.layers import Dense, MaxPooling2D
from keras.layers import Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from keras.utils.generic_utils import get_custom_objects
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.stats import stats
# Train whole data then test on decades
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# Importing sklearn libraries

epochs = 10
batch_size = 32
image_count = 100
limit_data = False
max_samples = 12000
test_run = False
generate_data = False
load_image_data = False
load_features = True
load_features_flat = False
show_dist_graphs = False
normalize = True

limit_memory = True

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
img_path = clean_data[3][3]
        # print(img_path)
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=1500)

def get_orb_features(dataSource):
    x_scratch = []
    for comic in dataSource:
        img_path = comic[3]
        # print(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # descriptors = sift.detectAndCompute(img, None)
        # descriptors = surf.detectAndCompute(img, None)
        descriptors = orb.detectAndCompute(img, None)
        x = np.vstack(descriptors)


    # features = pre_model.predict(x, batch_size=batch_size)
    # features_flatten = features.reshape((features.shape[0], 7 * 7 * 512))
    return x




img = cv2.drawKeypoints(img, keypoints_orb, None)
cv2.imshow("Image", img)
cv2.waitKey(0)

# img_data = image.img_to_array(img)
#         # img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
# img_data = np.expand_dims(img_data, axis=0)
# img_data = np.squeeze(img_data, axis=0)

if limit_data:
    clean_data = clean_data[:max_samples]
data_y = []
data = clean_data

def create_y(data):
    for comic in data:
        data_y.append(int(comic[6]))
    np.save("data_y.npy", data_y)
    np.savetxt('data_y.csv', data_y, delimiter=',')

vggmodel = VGG16( weights="imagenet",include_top=False)
    # model = VGG16( include_top=False)
vggmodel.summary()


if generate_data:
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
    features = np.load("features.npy")
if load_features_flat:
    print("Loading flattend feature data")
    features_flatten = np.load("features_flatten.npy")
data_y = np.load("data_y.npy")
if limit_data:
    data_y = data_y[:max_samples]



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
if test_run:
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
    if(use_orb):
        features = get_orb_features(data)
    else:
        train, test, val = features[training_idx,:], features[test_idx,:], features[val_idx,:]

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
    sns.distplot(y_pos)
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
    # plt.scatter(x_pos, y_pos, alpha=0.5, color='red')
    # plt.show()
    #
    # yPlot = test_y
    # y_pos = np.sort(yPlot.astype(int))
    # x_pos = np.arange(len(yPlot))
    # plt.scatter(x_pos, y_pos, alpha=0.5, color='blue')
    # plt.show()
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
    # EarlyStoppingByLossVal(monitor='val_loss', value=0.001, verbose=1),
    checkpointer]


optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
# optimizer = SGD(lr=0.01, momentum=0.9)

def baseline_model():
    model = Sequential()
    model.add(MaxPooling2D(pool_size=7, input_shape=train.shape[1:]))
    model.add(Flatten())
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))

    #     # model.add(Dense(256, activation=swish))
    #     # model.add(Dropout(0.5))
    #     # # model.add(Dense(1024, activation="relu"))
    #     # # model.add(Dropout(0.5))
    #     # # model.add(Dense(64, activation="relu"))
    #     # # model.add(Dropout(0.5))
    #     # # model.add(Dense(1, activation="relu"))
    #     # model.add(Dense(1))
    model.add(Dense(1024, activation=swish, use_bias=True))
    # model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model.add(Dropout(0.5))
    # model.add(Dense(1024, activation=swish, use_bias=True))
    # # model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    # model.add(Dropout(0.5))
    # model.add(Dense(256, activation=swish, use_bias=True))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'mae','mape'])
    return model


print("Shape:" + str(train.shape[1:]))


estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size= 100, verbose=False)

# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, train_features, y_train, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

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

plot_loss(history)

plt.scatter(test_y, prediction)
plt.xlabel("True Values")
plt.ylabel("Predictions")
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

print("Mean Error:" + str(mean_error))
print("Min Error:" + str(min_error))
print("Max Error:" + str(max_error))
print("Std Error:" + str(std_error))

