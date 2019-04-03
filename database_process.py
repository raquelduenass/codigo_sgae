# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:35:32 2018

@author: rds
"""
from __future__ import print_function
import os
from pydub import AudioSegment
import csv
import utils
import numpy as np
import librosa
from scipy.io import wavfile
# ♥


def data_and_labels(music_path, data_list, label_list, moments_list, label, separation):
    
    for music_file in os.listdir(music_path):
        entry_path = os.path.join(music_path, music_file)
        
        if music_file.endswith('.wav'):
            audio = AudioSegment.from_file(entry_path, format="wav")
            for i in range(int(audio.duration_seconds/separation)):
                data_list.append(os.path.abspath(os.path.join(music_path, music_file)))
                moments_list.append(i*separation)
                label_list.append(label)
                
        elif music_file.endswith('.mp3'):
            audio = AudioSegment.from_file(entry_path, format="mp3")
            for i in range(int(audio.duration_seconds/separation)):
                data_list.append(os.path.abspath(os.path.join(music_path, music_file)))
                moments_list.append(i*separation)
                label_list.append(label)
                
    return data_list, label_list, moments_list


# TODO: CORREGIR VARIOS AUDIOS
def data_and_labels_mu_speak(label, music_file, pre_label,
                             moments_list, start_list, end_list,
                             separation, overlap):
    audio = AudioSegment.from_file(music_file, format="mp3")
    if not(overlap == 0):
        param = separation / overlap
        start_list[:] = [int(x/overlap) for x in start_list]
        end_list[:] = [int(x/overlap) for x in end_list]
        labels = np.ones((int((int(audio.duration_seconds / overlap)//param)*param),))
        for i in range(int(audio.duration_seconds*param)):
            moments_list.append(i * overlap)
    else:
        start_list[:] = [int(x / separation) for x in start_list]
        end_list[:] = [int(x / separation) for x in end_list]
        labels = np.ones((int(audio.duration_seconds / separation),))
        for i in range(int(audio.duration_seconds / separation)):
            moments_list.append(i * separation)

    if label == 'M':
        labels = labels.astype(int)*3
        for i in range(len(start_list)):
            labels[start_list[i]:end_list[i]] = 0
        label_list = list(labels)

    else:
        for i in range(len(start_list)):
            for j in range(start_list[i], end_list[i]):
                if pre_label[j] == 0:
                    pre_label[j] = 1
                else:
                    pre_label[j] = 2
        label_list = list(pre_label)

    return label_list, moments_list


# TODO: CORREGIR VARIOS AUDIOS
def create_database(root_data_path, separated, separation, overlap):
    
    data_list, label_list, moments_list = [], [], []
    
    if separated:
        for classes in os.listdir(root_data_path):
            data_list, label_list, moments_list = data_and_labels(os.path.join(root_data_path, classes),
                                                                  data_list, label_list, moments_list,
                                                                  classes, separation)
        dct = {'music': 0, 'music_speech': 1, 'speech': 2, 'noise': 3}
        label_list = list(map(dct.get, label_list))

    else:
        start_music, end_music, start_speech, end_speech = [], [], [], []
        
        for csv_file in os.listdir(os.path.join(root_data_path, 'meta')):
            if csv_file.endswith('.csv'):
                audio_file = os.path.join(root_data_path, str(csv_file.split('.')[0])+'.wav')
                csv_file = os.path.join(root_data_path, 'meta', csv_file)
                data_list.append(os.path.abspath(audio_file))
                
                with open(csv_file, newline='') as csv_data:
                    info = []
                    spam_reader = csv.reader(csv_data, delimiter=' ', quotechar='|')
                    for row in spam_reader:
                        info.append(row)
                        if row[0].split(',')[2] == 'm':
                            start = int(round(float(row[0].split(',')[0])))
                            start_music.append(start)
                            end_music.append(start + int(round(float(row[0].split(',')[1]))))
                        elif row[0].split(',')[2] == 's':
                            start = int(round(float(row[0].split(',')[0])))
                            start_speech.append(start)
                            end_speech.append(start + int(round(float(row[0].split(',')[1]))))
                
                label_list, moments_list = data_and_labels_mu_speak(
                    'M', audio_file, label_list, moments_list,
                    start_music, end_music, separation, overlap)
                label_list, moments_list = data_and_labels_mu_speak(
                    'H', audio_file, label_list, moments_list,
                    start_speech, end_speech, separation, overlap)
    
    # Save audios, labels and moments in file
    utils.list_to_file(data_list, os.path.join(root_data_path, 'data.txt'))
    utils.list_to_file(label_list, os.path.join(root_data_path, 'labels.txt'))
    utils.list_to_file(moments_list, os.path.join(root_data_path, 'moments.txt'))
    return


def classes_combination(root_data_path, equal, combs, speech_pct):
    
    classes = os.listdir(root_data_path)
    speech_path = os.path.join(root_data_path, classes[combs[1]])
    music_path = os.path.join(root_data_path, classes[combs[0]])
    speech_files = [os.path.join(speech_path, i) for i in os.listdir(speech_path)]
    music_files = [os.path.join(music_path, i) for i in os.listdir(music_path)]
    j = 0
    for i in range(len(os.listdir(speech_path))):
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
                j = j+1
                if j >= len(music_files):
                    j = 0
            music = music[0:len(speech)]
        
        comb = speech_pct*speech+(1-speech_pct)*music
        if combs[0] == 0:
            folder = 'music_speech'
        else:  # if combs[0] == 2:
            folder = 'speech_noise'
        output_path = os.path.join(root_data_path, folder, 'comb_'+str(i)+'.wav')
        wavfile.write(output_path, sr_music, comb)
    return


def compute_mel_gram(src, sr, power, duration):

    # mel-spectrogram parameters
    n_fft = 512
    n_mel = 96
    hop_len = 256
    n_sample = src.shape[0]
    n_sample_fit = int(duration*sr)

    if n_sample < n_sample_fit:  # if too short
        src = np.concatenate([src, np.zeros((int(duration*sr) - n_sample,))])
    elif n_sample > n_sample_fit:  # if too long
        src = src[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]
    mel = librosa.feature.melspectrogram(
            y=src, sr=sr, hop_length=hop_len,
            n_fft=n_fft, n_mels=n_mel, power=power)
    ret = librosa.power_to_db(mel)
    return ret


def extract_spec_grams(data_path, save_path):
    separation = 2
    power = 2
    for classes in os.listdir(data_path):
        j = 0
        for files in os.listdir(os.path.join(data_path, classes)):
            file_path = os.path.join(data_path, classes, files)
            if file_path.endswith('.wav') or file_path.endswith('.mp3'):
                audio, sr = librosa.load(file_path)
                for i in range(0, int(librosa.get_duration(audio)), separation):
                    segment = audio[i*sr:(i+separation)*sr]
                    mel = compute_mel_gram(segment, sr, power, separation)
                    np.save(os.path.join(save_path, classes, 'mel_'+str(j)+'.npy'), mel)
                    j = j+1


def data_files(data_path):
    file_names, labels = [], []
    for classes in os.listdir(data_path):
        class_path = os.path.join(data_path, classes)
        for files in os.listdir(class_path):
            file_names.append(os.path.join(class_path, files))
            labels.append(classes)

    utils.list_to_file(file_names, os.path.join(data_path, 'data.txt'))
    utils.list_to_file(labels, os.path.join(data_path, 'labels.txt'))
    return


# classes_combination('C:/Users/rds/Documents/GitHub/data_sgae/', False, combs, 0.9)
create_database('C:/Users/rds/Documents/GitHub/data_sgae/muspeak', False, 2, 0.5)
