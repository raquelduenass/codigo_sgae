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

# Data directories
root_data_path = "./data/music_speech"
root_data_path2 = "./data/muspeak"

# Folder exploration
classes = os.listdir(root_data_path)

##♥
def data_and_labels(music_path, data_list, label_list, moments_list, label):
    
    for music_file in os.listdir(music_path):
        if music_file.endswith('.wav'):
            entry_path = os.path.join(music_path, music_file)
            sr, audio = wavfile.read(entry_path)
            
            for i in range(int(len(audio)/sr)):
                data_list.append(os.path.abspath(os.path.join(music_path, music_file)))
                moments_list.append(i)
                label_list.append(label)
                
    return data_list, label_list, moments_list


def data_and_labels_muspeak(music_file, data_list, label_list, moments_list, start_list, end_list):
    
    audio = AudioSegment.from_file(music_file, format="mp3")
    labels = np.ones((int(audio.duration_seconds),))
    
    for i in range(len(start_list)):
        labels[start_list[i]:end_list[i]] = 0
    for i in range(int(audio.duration_seconds)):
        data_list.append(os.path.abspath(music_file))
        moments_list.append(i)
       
    label_list = list(labels)
    return data_list, label_list, moments_list


def create_database(root_data_path, separated):
    
    data_list = []
    label_list = []
    moments_list = []
    
    if separated:
        for classes in os.listdir(root_data_path):
            data_list, label_list, moments_list = data_and_labels(os.path.join(root_data_path, classes),
                                                data_list, label_list, moments_list, classes)
    else:
        start_list = []
        end_list = []
        
        for csv_file in os.listdir(os.path.join(root_data_path,'meta')):
            if csv_file.endswith('.csv'):
                audio_file = csv_file.split('.')[0]+'.wav'
                audio_file = os.path.join(root_data_path,audio_file)
                csv_file = os.path.join(root_data_path,'meta',csv_file)
                
                with open(csv_file, newline='') as csvfile:
                    info = []
                    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                    for row in spamreader:
                        info.append(row)
                        if row[0].split(',')[2] == 'm':
                            start = int(round(float(row[0].split(',')[0])))
                            start_list.append(start)
                            end_list.append(start + int(round(float(row[0].split(',')[1]))))
                
                data_list, label_list, moments_list = data_and_labels_muspeak(
                        audio_file, data_list, label_list, moments_list,
                        start_list, end_list)
        
    le = preprocessing.LabelEncoder()
    le.fit(label_list)
    label_list = list(le.transform(label_list))
    
    # Save audios, labels and moments in file
    utils.list_to_file(data_list, os.path.join(root_data_path,'data.txt'))
    utils.list_to_file(label_list, os.path.join(root_data_path,'labels.txt'))
    utils.list_to_file(moments_list, os.path.join(root_data_path,'moments.txt'))
    
    return

def classes_combination(root_data_path):
    
    classes = os.listdir(root_data_path)
    speech_path = os.path.join(root_data_path, classes[2])
    music_path = os.path.join(root_data_path, classes[1])
    speech_files = [os.path.join(speech_path, i) for i in os.listdir(speech_path)]
    music_files = [os.path.join(music_path, i) for i in os.listdir(music_path)]
    
    for i in range(len(os.listdir(music_path))):
        speech, sr_speech = librosa.load(speech_files[i])
        music, sr_music = librosa.load(music_files[i])
        comb = 0.6*speech+0.4*music
        comb_scaled = comb/np.max(np.abs(comb))
        output_path = os.path.join(root_data_path, 'music_speech_wav',
                                   'comb_'+str(i)+'.wav')
        wavfile.write(output_path, sr_music, comb_scaled)
        #librosa.output.write_wav(output_path, comb_scaled, sr_music)
        
    return

#classes_combination(root_data_path)
#create_database(root_data_path, True)
create_database(root_data_path2, False)