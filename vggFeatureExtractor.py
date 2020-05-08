import warnings

from keras.applications import VGG16
from keras.preprocessing import image
from keras_applications import imagenet_utils
from pandas import np

batch_size = 32
data_y = []
def create_y(data):
    for comic in data:
        data_y.append(int(comic[6]))
    np.save("data_y.npy", data_y)
    np.savetxt('data_y.csv', data_y, delimiter=',')

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




#Load up the clean data
clean_data = np.load('cleanData.npy',allow_pickle=True)
model = VGG16( weights="imagenet",include_top=False)

create_y(clean_data)

model.summary()
print("Creating Features")
image_data, features, features_flatten = create_features(clean_data, model)

np.save("image_data.npy", image_data)
np.save("features.npy", features)
np.save("features_flatten.npy", features_flatten)

