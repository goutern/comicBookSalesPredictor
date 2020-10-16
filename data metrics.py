import numpy as np

clean_data = np.load('data_y.npy',allow_pickle=True)

mean_error = np.mean(clean_data)
min_error = np.min(clean_data)
max_error = np.max(clean_data)
std_error = np.std(clean_data)



print("Mean Error:" + str(mean_error))
print("Min Error:" + str(min_error))
print("Max Error:" + str(max_error))
print("Std Error:" + str(std_error))