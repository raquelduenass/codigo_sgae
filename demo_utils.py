import numpy as np
import librosa

#from pydub import AudioSegment
#from pyAudioAnalysis import audioFeatureExtraction
#
#def create_spectrogram(i, moments, data, target_size):
#    newAudio = AudioSegment.from_file(data[i], format="wav")
#    segment = newAudio[int(moments[i])*1000:(int(moments[i])+1)*1000]
#
#    Fs=20000
#    segment = segment.get_array_of_samples()
#    pxx, TimeAxis, FreqAxis = audioFeatureExtraction.stSpectogram(
#            segment, Fs, round(Fs * 0.010), round(Fs * 0.020), False)
#    pxx = np.resize(pxx, target_size)
#    return pxx


def compute_melgram(src):
    ''' Compute a mel-spectrogram and returns it in a shape of (96,1366), where
    96 == #mel-bins and 1366 == #time frame'''

    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 1  # to make it 1366 frame..
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.concatenate([src, np.zeros((int(DURA*SR) - n_sample,))])
    elif n_sample > n_sample_fit:  # if too long
        src = src[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]
    ret = librosa.amplitude_to_db(librosa.feature.melspectrogram(
            y=src, sr=SR, hop_length=HOP_LEN,
            n_fft=N_FFT, n_mels=N_MELS)**2,
            ref_power=1.0)
    return ret


def silence_detection(audio_slice):
    silence_thresh = -16
    silence = librosa.feature.rmse(audio_slice) <= silence_thresh
    return silence