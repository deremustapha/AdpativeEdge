import numpy as np
from scipy.signal import butter, iirnotch, filtfilt

class EMGPreprocessing:
    def __init__(self, fs=250, notch_freq=60.0, low_cut=10.0, high_cut=120.0, quality_factor=30.0, order=4):
        self.fs = fs
        self.notch_freq = notch_freq
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.quality_factor = quality_factor
        self.order = order

    def remove_mains(self, data: np.ndarray) -> np.ndarray:
        """
        Apply a notch filter to remove the main frequency interference from the data.

        Parameters:
        data (ndarray): The input data array to be filtered. Shape (n_samples, n_channels).

        Returns:
        ndarray: The filtered data with the main frequency interference removed.
        """
        b, a = iirnotch(self.notch_freq, self.quality_factor, self.fs)
        return filtfilt(b, a, data, axis=1)

    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply a bandpass filter to the given data.

        Parameters:
        data (ndarray): The input data to be filtered. It is assumed to be a 2D array where filtering is applied along the second axis (axis=1).
        lowcut (float): The lower frequency cutoff for the bandpass filter.
        highcut (float): The upper frequency cutoff for the bandpass filter.

        Returns:
        ndarray: The filtered data.
        """
        nyquist = 0.5 * self.fs
        low = self.low_cut / nyquist
        high = self.high_cut / nyquist
        b, a = butter(self.order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=1)
    
    
    def highpass_filter(self, data):

        nyquist = 0.5 * self.fs
        normalized_cutoff = self.low_cut / nyquist
        b, a = butter(self.order, normalized_cutoff, btype='high')
        
        return filtfilt(b, a, data, axis=1)