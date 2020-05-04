
import numpy as np
import cv2

clean_data = np.load('cleanData.npy',allow_pickle=True)

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb_feature_count = 1000

orb = cv2.ORB_create(nfeatures=orb_feature_count, scoreType=cv2.ORB_FAST_SCORE)

data = clean_data

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



print("getting Orb")
split_data = np.array_split(data, 10)
for x in range(0 , 10):
    try:
        print("Set: " + str(x))
        np.save("features_orb_"+ str(x) + ".npy", get_orb_features(split_data[x]))
    except:
        continue
np.save("features_orb.npy", get_orb_features(data))
print("Orb Saved")