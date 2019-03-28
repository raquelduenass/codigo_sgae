# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:35:32 2018

@author: rds
"""
from __future__ import print_function
import os
from pydub import AudioSegment
from sklearn import preprocessing
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


def data_and_labels_mu_speak(music_file, data_list, moments_list, start_list, end_list, separation):
    
    audio = AudioSegment.from_file(music_file, format="mp3")
    labels = np.ones((int(audio.duration_seconds/separation),))
    start_list[:] = [int(x/separation) for x in start_list]
    end_list[:] = [int(x/separation) for x in end_list]
    
    for i in range(len(start_list)):
        labels[start_list[i]:end_list[i]] = 0
    for i in range(int(audio.duration_seconds/separation)):
        data_list.append(os.path.abspath(music_file))
        moments_list.append(i*separation)
       
    label_list = list(labels)
    return data_list, label_list, moments_list


def create_database(root_data_path, separated, separation):
    
    data_list = []
    label_list = []
    moments_list = []
    
    if separated:
        for classes in os.listdir(root_data_path):
            data_list, label_list, moments_list = data_and_labels(os.path.join(root_data_path, classes),
                                                                  data_list, label_list, moments_list, classes,
                                                                  separation)
    else:
        start_list = []
        end_list = []
        
        for csv_file in os.listdir(os.path.join(root_data_path, 'meta')):
            if csv_file.endswith('.csv'):
                audio_file = os.path.join(root_data_path, str(csv_file.split('.')[0])+'.wav')
                csv_file = os.path.join(root_data_path, 'meta', csv_file)
                
                with open(csv_file, newline='') as csv_data:
                    info = []
                    spam_reader = csv.reader(csv_data, delimiter=' ', quotechar='|')
                    for row in spam_reader:
                        info.append(row)
                        if row[0].split(',')[2] == 'm':
                            start = int(round(float(row[0].split(',')[0])))
                            start_list.append(start)
                            end_list.append(start + int(round(float(row[0].split(',')[1]))))
                
                data_list, label_list, moments_list = data_and_labels_mu_speak(
                        audio_file, data_list, moments_list,
                        start_list, end_list, separation)
        
    le = preprocessing.LabelEncoder()
    le.fit(label_list)
    label_list = list(le.transform(label_list))
    
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


def labels_demo(data_path, file, num_classes):
    labels_old = utils.file_to_list(os.path.join(data_path, file), False)
    dct = {'1\n': '0', '0\n': '3'}
    # if num_classes == 3:
    #     dct = {'1\n': '2'}
    # else:
    #     dct = {'1\n': '3'}
    labels = list(map(dct.get, labels_old))
    utils.list_to_file(labels, os.path.join(data_path, 'labels'+str(num_classes)+'.txt'))
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
    file_names = []
    labels = []
    for classes in os.listdir(data_path):
        class_path = os.path.join(data_path, classes)
        for files in os.listdir(class_path):
            file_names.append(os.path.join(class_path, files))
            labels.append(classes)

    utils.list_to_file(file_names, os.path.join(data_path, 'data.txt'))
    utils.list_to_file(labels, os.path.join(data_path, 'labels.txt'))
    return


def data_rename(data_path):
    for classes in os.listdir(data_path):
        i = 0
        class_path = os.path.join(data_path, classes)
        for files in os.listdir(class_path):
            os.rename(os.path.abspath(files), os.path.join(class_path, 'mel_'+f'{i:6}'+'.png'))
            i = i + 1
    return
