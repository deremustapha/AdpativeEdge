import pywt
from scipy.signal import stft
import numpy as np

def get_features(data, fs=250, n_samples=40):
   
    f, t, cstft = stft(data, fs, nperseg=n_samples)
    return np.abs(cstft)


def get_stft_features(data, samples=40):
   
    cstft_tr = []
   
    for i, idx in enumerate(data):
        a = get_features(data=data[i], fs=250, n_samples=samples)
        cstft_tr.append(a)
   
    cstft_x_train = np.array(cstft_tr)
    cstft_x_train = cstft_x_train.transpose(0, 3, 1, 2)
    return cstft_x_train


def get_cwt_features(data, wavelet='mexh', scales=range(1, 33)):
   
    cwt_tr = []
   
    for i, idx in enumerate(data):
        a,_ = pywt.cwt(data[i], scales, wavelet)
        cwt_tr.append(a)
   
    cwt_x_train = np.array(cwt_tr)
    return cwt_x_train


def get_raw_data(data):
    data = np.expand_dims(data, axis=1)
    return data

# window_train_data = get_cwt_features(window_train_data)
# window_test_data = get_cwt_features(window_test_data)

# window_train_data = stft_image(window_train_data)
# window_train_data = window_train_data.transpose(0, 3, 1, 2) #.reshape(window_train_data.shape[0], window_train_data.shape[2], window_train_data.shape[1] * window_train_data.shape[3])
# window_test_data = stft_image(window_test_data)
# window_test_data = window_test_data.transpose(0, 3, 1, 2) # .reshape(window_test_data.shape[0], window_test_data.shape[2], window_test_data.shape[1] * window_test_data.shape[3])
# window_train_data.shape, window_train_labels.shape, window_test_data.shape, window_test_labels.shape


# window_train_data = np.expand_dims(window_train_data, axis=1)
# window_test_data = np.expand_dims(window_test_data, axis=1)
# window_train_data.shape, window_train_labels.shape, window_test_data.shape, window_test_labels.shape