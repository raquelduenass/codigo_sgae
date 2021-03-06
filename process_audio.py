import librosa
import numpy as np


def fade_in_out(segment, sr_music):
    """

    :param segment: audio sequence
    :param sr_music: sample rate of the audios
    :return fade: fade-in transition
    :return rest: merge of the no fading and fade-out transition sequences
    """
    fade_length = int(2 * sr_music)
    factor = 0
    separations = 10
    n_samples = int(librosa.get_duration(segment) * sr_music)
    fade = segment[0:int(fade_length / separations)] * factor
    for i in range(separations - 1):
        fade = np.append(fade, segment[int((i + 1) * fade_length / separations):
                                       int((i + 2) * fade_length / separations)] * factor)
        fade = fade[0:fade_length]
        factor += 1 / separations
    rest = segment[fade_length:n_samples - fade_length]
    for i in range(separations):
        rest = np.append(fade, segment[int(n_samples - (separations - i) * fade_length / 10):
                                       int(n_samples - (separations - i - 1) * fade_length / 10)] * factor)
        factor -= 1 / separations

    return fade, rest


def separate_many_audio(self, index_array):
    """

    :param self:
    :param index_array:
    :return segments:
    """
    segments = []
    actual_file = 0

    # Identify audio to classify
    for j in range(-len(self.files_length) + 1, 1):
        if index_array[0] < self.files_length[-j]:
            actual_file = -j

    # Select limits of audio segment
    offset = index_array[0]
    if not (actual_file == 0):
        offset = offset - self.files_length[actual_file - 1]
    if not (self.overlap == 0):
        offset = offset * self.overlap
        duration = (self.batch_size - 1) * self.overlap + self.separation
    else:
        offset = offset * self.separation
        duration = self.batch_size * self.separation

    # Load and re-sample audio segment
    audio, sr_old = librosa.load(self.file_names[actual_file],
                                 offset=offset, duration=duration)
    audio = librosa.resample(audio, sr_old, self.sr)
    real_duration = librosa.get_duration(audio)

    minus = 0
    for j in range(len(index_array)):
        if self.overlap == 0:  # No overlapping
            segments.append(audio[j * self.separation * self.sr:
                                  (j + 1) * self.separation * self.sr])
        else:
            if (j - minus) * self.overlap + self.separation <= real_duration:
                segments.append(audio[int((j - minus) * self.overlap * self.sr):
                                      int(((j - minus) * self.overlap + self.separation) * self.sr)])
            else:  # Change of audio file
                minus = j
                actual_file = actual_file + 1
                audio, sr_old = librosa.load(self.file_names[actual_file],
                                             duration=(self.batch_size - j) * self.overlap + self.separation)
                audio = librosa.resample(audio, sr_old, self.sr)
                real_duration = librosa.get_duration(audio)
                segments.append(audio[0:int(self.separation * self.sr)])

    return segments


def compute_mel_gram(separation, sr, power, segment):
    """
    Computation of the mel-spectrogram of an audio sequence
    :param separation: time duration represented in the spectrogram
    :param sr: sample rate of the audio
    :param power: representation of the spectrogram: energy/power
    :param segment: audio fragment from which extracting the mel-spectrogram
    :return ret: mel-spectrogram
    """
    n_fft = 512
    n_mel = 96
    hop_len = 256
    n_sample = segment.shape[0]
    n_sample_fit = int(separation*sr)

    if n_sample < n_sample_fit:  # if too short
        src = np.concatenate([segment, np.zeros((int(separation*sr) - n_sample,))])
    elif n_sample > n_sample_fit:  # if too long
        src = segment[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]
    else:
        src = segment
    mel = librosa.feature.melspectrogram(
            y=src, sr=sr, hop_length=hop_len,
            n_fft=n_fft, n_mels=n_mel, power=power)

    ret = librosa.power_to_db(mel)
    return ret


def silence_detection(audio_slice):
    """

    :param audio_slice:
    :return silence:
    """
    silence_thresh = -16
    silence = librosa.feature.rmse(audio_slice) <= silence_thresh
    return silence
