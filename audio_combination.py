
import os
import librosa
import numpy as np
from scipy.io import wavfile


def classes_combination(root_data_path, equal, speech_pct):

    classes = os.listdir(root_data_path)
    speech_path = os.path.join(root_data_path, classes[1])
    music_path = os.path.join(root_data_path, classes[0])
    speech_files = [os.path.join(speech_path, i) for i in os.listdir(speech_path)]
    music_files = [os.path.join(music_path, i) for i in os.listdir(music_path)]
    folder = 'music_speech'
    j = 0
    for i in range(len(os.listdir(speech_path))):
        print(i)
        speech, sr_speech = librosa.load(speech_files[i])
        music, sr_music = librosa.load(music_files[j])
        speech = librosa.resample(speech, sr_speech, sr_music)

        if equal:
            j = j+1
        else:
            while len(music) < len(speech):
                add_music, sr_add_music = librosa.load(music_files[j])
                if not sr_music == sr_add_music:
                    add_music = librosa.resample(add_music, sr_add_music, sr_music)
                music = np.append(music, add_music)
                j = j + 1
                if j >= len(music_files):
                    j = 0
            music = music[0:len(speech)]

        comb = speech_pct * speech + (1 - speech_pct) * music
        output_path = os.path.join(root_data_path, folder, 'comb_' + str(i) + '.wav')
        wavfile.write(output_path, sr_music, comb)
    return
