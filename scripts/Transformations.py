from scipy.signal import stft, cwt, ricker, morlet2
import numpy as np

FREQ = 400
class Transformations:
    #Each ecg is a 4096x12 matrix
    def __init__(self, ecgs):
        self.ecgs = ecgs
        self.n_cases = ecgs.shape[0]


    def get_cwt_arrays(self, wavelet = ricker, scales = np.linspace(1,128,100)):
        # We calculate the dimensions of the output matrix using the first input matrix  
        if wavelet == ricker:
            f = FREQ / (2*np.pi*scales)
        elif wavelet == morlet2:
            f = (5 * FREQ) / (2 * np.pi * scales)
        else:
            raise NotImplementedError("Wavelet not recognized")

        t = np.arange(self.ecgs[0].shape[0]) / FREQ
        return f,t

    def cwt(self, wavelet = ricker, scales = np.linspace(1,128, 100)):  
        cwts = []
        for i in range(self.n_cases):
            ecg = self.ecgs[i]
            cwt_i = np.zeros((len(scales) * ecg.shape[1], ecg.shape[0]))
            for j in range(ecg.shape[1]): #Iterating over the derivations
                Zxx = np.abs(cwt(ecg[:,j], wavelet=wavelet, widths=scales))
                cwt_i[j * len(scales):(j + 1) * len(scales), :] = Zxx
            cwts.append(cwt_i)
        return np.array(cwts)

    def get_stft_arrays(self, nperseg=256, noverlap=128):
        # We calculate the dimensions of the output matrix using the first input matrix  
        f, t, _ = stft(self.ecgs[0][:, 0], fs=FREQ, nperseg=nperseg, noverlap=noverlap)
        return f, t

    #Function that performs the STFT transform
    def stft(self, nperseg=256, noverlap=128):
        # We calculate the dimensions of the output matrix using the first input matrix
        n_frequencies, n_times = stft(self.ecgs[0][:, 0], fs=FREQ, nperseg=nperseg, noverlap=noverlap)[2].shape
        stfts = []
        #Iterating over the data
        for i in range(self.n_cases):
            ecg = self.ecgs[i]
            stft_i = np.zeros((n_frequencies * ecg.shape[1], n_times))
            for j in range(ecg.shape[1]):
                _, _, Zxx = stft(ecg[:, j], fs=FREQ, nperseg=nperseg, noverlap=noverlap)
                stft_i[j * n_frequencies:(j + 1) * n_frequencies, :] = np.abs(Zxx)
            stfts.append(stft_i)
        return np.array(stfts)
            