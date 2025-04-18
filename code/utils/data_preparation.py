import numpy as np
import os
from scipy.io import loadmat
from typing import List, Dict, Tuple
from torch.utils.data import Dataset

class EMGDataPreparation:
    """
    A class used to prepare EMG data for machine learning tasks.

    Attributes
    ----------
    base_path : str
        The base directory path where the EMG data is stored.
    fs : int
        The sampling frequency of the EMG data.
    rec_time : float
        The total duration of the EMG data in seconds.
    total_time : int
        The total number of samples in the EMG data.

    Methods
    -------
    get_per_subject_file(subject_number: int, num_gesture: int, train_repetition: List[int], test_repetition: List[int]) -> Tuple[str, Dict[str, List[str]], Dict[str, List[str]]]:
        Retrieves the file paths for training and testing data for a specific subject and gesture.

    load_data_per_subject(path: str, selected_gesture: List[int], train_gesture: Dict[str, List[str]], test_gesture: Dict[str, List[str]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        Loads the EMG data for a specific subject and gesture from the given file paths.

    get_data_labels(data_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        Concatenates the data and generates corresponding labels.

    get_max_label(label: np.ndarray) -> int:
        Determines the most frequent label in the given array.

    window_with_overlap(data: np.ndarray, label: np.ndarray, window_time: int = 200, overlap: int = 60, no_channel: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        Segments the data into overlapping windows and assigns labels to each window.
    """
    def __init__(self, base_path: str, fs: int, rec_time: float):
        self.base_path = base_path
        self.fs = fs
        self.rec_time = rec_time
        self.total_time = int(rec_time * fs)

    def get_per_subject_file(self, subject_number: int, num_gesture: int, activate_session: bool, session: int,
                             train_repetition: List[int], test_repetition: List[int]) -> Tuple[str, Dict[str, List[str]], Dict[str, List[str]]]:
        
        if activate_session == False:
            base_path = os.path.join(self.base_path, f"subject_{subject_number}")
        
        else:
            base_path = os.path.join(self.base_path, f"subject_{subject_number}",  f"session_{session}")
        data_file = os.listdir(base_path)

        train_ges = {str(i): [] for i in range(num_gesture)}
        test_ges = {str(i): [] for i in range(num_gesture)}

        for i in range(num_gesture):
            gesture_files = [file for file in data_file if file[7] == str(i + 1) or file[8] == str(i + 1)]
            train_ges[str(i)] = [gesture_files[rep - 1] for rep in train_repetition]

            ##  Work on this code later to improve if test repetition is not in the data
            test_ges[str(i)] = [gesture_files[rep - 1] for rep in test_repetition]

        return base_path, train_ges, test_ges

    def load_data_per_subject(self, path: str, selected_gesture: List[int],
                              train_gesture: Dict[str, List[str]], test_gesture: Dict[str, List[str]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        def load_gesture_data(gesture_files: List[str]) -> np.ndarray:
            return np.concatenate([loadmat(os.path.join(path, file))['data'][20:self.total_time, :].T for file in gesture_files], axis=1)

        train_data = {str(i - 1): load_gesture_data(train_gesture[str(i - 1)]) for i in selected_gesture}
        test_data = {str(i - 1): load_gesture_data(test_gesture[str(i - 1)]) for i in selected_gesture}

        return train_data, test_data

    @staticmethod
    def get_data_labels(data_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        data = np.concatenate([data_dict[key] for key in data_dict], axis=1)
        label = np.concatenate([np.full(data_dict[key].shape[1], int(key)) for key in data_dict])
        return data, label

    @staticmethod
    def get_max_label(label: np.ndarray) -> int:
        return np.argmax(np.bincount(label))

    def window_with_overlap(self, data: np.ndarray, label: np.ndarray, window_time: int = 200,
                            overlap: int = 60, no_channel: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        samples = int(self.fs * (window_time / 1000))
        num_overlap_samples = int(samples * (overlap / 100))
        step_size = samples - num_overlap_samples
        num_windows = (data.shape[1] - samples) // step_size + 1

        data_matrix = np.array([data[:no_channel, i*step_size:i*step_size+samples] for i in range(num_windows)])
        label_vectors = np.array([self.get_max_label(label[i*step_size:i*step_size+samples]) for i in range(num_windows)])

        return data_matrix, label_vectors



def shuffle_data(data, labels):
    """
    Shuffle the data and corresponding labels.

    Parameters:
    data (numpy.ndarray): The data to be shuffled.
    labels (numpy.ndarray): The labels corresponding to the data.

    Returns:
    tuple: A tuple containing the shuffled data and labels.
    """
    idx = np.random.permutation(len(data))
    return data[idx], labels[idx]


def data_split(data: np.ndarray, label: np.ndarray, train_percent: float = 80) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_ratio = train_percent / 100
    train_size = int(data.shape[0] * train_ratio)
    return data[:train_size], label[:train_size], data[train_size:], label[train_size:]



class EMGDataset(Dataset):
    """
    A custom dataset class for Electromyography (EMG) data.

    Attributes:
        data (array-like): The EMG data samples.
        label (array-like): The labels corresponding to the EMG data samples.

    Methods:
        __len__():
            Returns the number of samples in the dataset.

        __getitem__(idx):
            Retrieves the data sample and its corresponding label at the specified index.

    Args:
        data (array-like): The EMG data samples.
        label (array-like): The labels corresponding to the EMG data samples.
    """

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


# def add_gaussian_noise(data, noise_level):
#     noise = np.random.normal(0, noise_level, data.shape)
#     return data + noise

# noise_levels = [0.1, 0.2, 0.3, 0.4]

# augmented_data = []
# augmented_labels = []

# for noise_level in noise_levels:
#     noisy_data = add_gaussian_noise(window_train_data, noise_level)
#     augmented_data.append(noisy_data)
#     augmented_labels.append(window_train_labels)

# augmented_data = np.concatenate(augmented_data, axis=0)
# augmented_labels = np.concatenate(augmented_labels, axis=0)

# window_train_data = np.concatenate((window_train_data, augmented_data), axis=0)
# window_train_labels = np.concatenate((window_train_labels, augmented_labels), axis=0)

# window_train_data.shape, window_train_labels.shape

def LOSO_CV(path, fs, record_time, train_test_splits, session_number, activate_session,
                          num_gesture, selected_gesture, train_repetition, test_repetition):
    """
    Loads and prepares EMG data for multiple subjects.
    This function iterates over a range of subjects, loads their respective EMG data,
    and concatenates the data into training and testing sets.
    Args:
        start_subject (int): The starting subject number (inclusive).
        end_subject (int): The ending subject number (inclusive).
    Returns:
        tuple: A tuple containing four elements:
            - train_X (np.ndarray): The concatenated training data for all subjects.
            - train_Y (np.ndarray): The concatenated training labels for all subjects.
            - test_X (np.ndarray): The concatenated testing data for all subjects.
            - test_Y (np.ndarray): The concatenated testing labels for all subjects.
    """

    emg_prep = EMGDataPreparation(base_path=path, fs=fs, rec_time=record_time)
    

    
    
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    val_labels, val_labels = [], []

    for train_test in train_test_splits:

            # Load Train Data
        load_train = 'train'
        train_list  = train_test_splits[train_test][load_train]

        for subject in train_list:
            print(f"Loading training data for subject {subject}...")
            subject_path, train_gesture, test_gesture = emg_prep.get_per_subject_file(subject_number=subject, num_gesture=num_gesture, 
                            session=session_number, activate_session=activate_session, 
                            train_repetition=train_repetition, test_repetition=test_repetition)
                        
            train_data_temp, test_data_temp = emg_prep.load_data_per_subject(
                subject_path, selected_gesture=selected_gesture, train_gesture=train_gesture, test_gesture=test_gesture)

            train_data_temp, train_labels_temp = emg_prep.get_data_labels(train_data_temp)
            test_data_temp, test_labels_temp = emg_prep.get_data_labels(test_data_temp)

        # print(f'train_data_temp shape: {train_data_temp.shape}')
        # print(f'train_labels_temp shape: {train_labels_temp.shape}')
        # print(f'test_data_temp shape: {test_data_temp.shape}')
        # print(f'test_labels_temp shape: {test_labels_temp.shape}')

            train_data.append(train_data_temp)
            test_data.append(test_data_temp)
            train_labels.append(train_labels_temp)
            test_labels.append(test_labels_temp)

        train_X = np.concatenate(train_data, axis=1)
        train_Y = np.concatenate(train_labels, axis=0)
        val_X = np.concatenate(test_data, axis=1)
        val_Y = np.concatenate(test_labels, axis=0)

        load_test = 'test'
        test_subject  = train_test_splits[train_test][load_test][0]
        print(f"Loading testing data for subject {test_subject}...")
        subject_path, train_gesture, test_gesture = emg_prep.get_per_subject_file(subject_number=test_subject, num_gesture=num_gesture, 
                session=session_number, activate_session=activate_session, 
                train_repetition=train_repetition, test_repetition=test_repetition)
        
        train_data_temp, test_data_temp = emg_prep.load_data_per_subject(
        subject_path, selected_gesture=selected_gesture, train_gesture=train_gesture, test_gesture=test_gesture)

        test_X, test_Y = emg_prep.get_data_labels(train_data_temp)

        return train_X, train_Y, val_X, val_Y, test_X, test_Y



def LOSO(path, fs, record_time, train_list, test_subject, session_number, activate_session,
                          num_gesture, selected_gesture, train_repetition, test_repetition):
    """
    Loads and prepares EMG data for multiple subjects.
    This function iterates over a range of subjects, loads their respective EMG data,
    and concatenates the data into training and testing sets.
    Args:
        start_subject (int): The starting subject number (inclusive).
        end_subject (int): The ending subject number (inclusive).
    Returns:
        tuple: A tuple containing four elements:
            - train_X (np.ndarray): The concatenated training data for all subjects.
            - train_Y (np.ndarray): The concatenated training labels for all subjects.
            - test_X (np.ndarray): The concatenated testing data for all subjects.
            - test_Y (np.ndarray): The concatenated testing labels for all subjects.
    """

    emg_prep = EMGDataPreparation(base_path=path, fs=fs, rec_time=record_time)
    

    
    
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    val_labels, val_labels = [], []



    for subject in train_list:
        print(f"Loading training data for subject {subject}...")
        subject_path, train_gesture, test_gesture = emg_prep.get_per_subject_file(subject_number=subject, num_gesture=num_gesture, 
                        session=session_number, activate_session=activate_session, 
                        train_repetition=train_repetition, test_repetition=test_repetition)
                    
        train_data_temp, test_data_temp = emg_prep.load_data_per_subject(
            subject_path, selected_gesture=selected_gesture, train_gesture=train_gesture, test_gesture=test_gesture)

        train_data_temp, train_labels_temp = emg_prep.get_data_labels(train_data_temp)
        test_data_temp, test_labels_temp = emg_prep.get_data_labels(test_data_temp)

        # print(f'train_data_temp shape: {train_data_temp.shape}')
        # print(f'train_labels_temp shape: {train_labels_temp.shape}')
        # print(f'test_data_temp shape: {test_data_temp.shape}')
        # print(f'test_labels_temp shape: {test_labels_temp.shape}')

        train_data.append(train_data_temp)
        test_data.append(test_data_temp)
        train_labels.append(train_labels_temp)
        test_labels.append(test_labels_temp)

    train_X = np.concatenate(train_data, axis=1)
    train_Y = np.concatenate(train_labels, axis=0)
    val_X = np.concatenate(test_data, axis=1)
    val_Y = np.concatenate(test_labels, axis=0)


    print(f"Loading testing data for subject {test_subject}...")
    subject_path, train_gesture, test_gesture = emg_prep.get_per_subject_file(subject_number=test_subject, num_gesture=num_gesture, 
            session=session_number, activate_session=activate_session, 
            train_repetition=train_repetition, test_repetition=test_repetition)
    
    train_data_temp, test_data_temp = emg_prep.load_data_per_subject(
    subject_path, selected_gesture=selected_gesture, train_gesture=train_gesture, test_gesture=test_gesture)

    test_X, test_Y = emg_prep.get_data_labels(train_data_temp)

    return train_X, train_Y, val_X, val_Y, test_X, test_Y

# Usage example:
# emg_prep = EMGDataPreparation(base_path="/path/to/data", fs=200, T=5)
# subject_path, train_ges, test_ges = emg_prep.get_per_subject_file(subject_number=1, num_gesture=5, train_repetition=[1, 2, 3], test_repetition=[4, 5])
# train_data, test_data = emg_prep.load_data_per_subject(subject_path, selected_gesture=[1, 2, 3, 4, 5], train_gesture=train_ges, test_gesture=test_ges)
# data, labels = emg_prep.get_data_labels(train_data)
# windowed_data, windowed_labels = emg_prep.window_with_overlap(data, labels)
# dataset = EMGDataset(windowed_data, windowed_labels)


# train_data, train_labels, test_data, test_labels = load_multiple_subject(start_subject=1, end_subject=1)
# preprocess = EMGPreprocessing(fs=200, notch_freq=60.0, low_cut=5.0, high_cut=99.0, order=4)
# train_data = preprocess.remove_mains(train_data)
# test_data = preprocess.remove_mains(test_data)

# train_data = preprocess.bandpass_filter(train_data)
# test_data = preprocess.bandpass_filter(test_data)

# train_data.shape, train_labels.shape, test_data.shape, test_labels.shape