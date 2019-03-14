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

##♥
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


def data_and_labels_muspeak(music_file, data_list, label_list, moments_list, start_list, end_list, separation):
    
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
    
    classes = os.listdir(root_data_path)
    
    if separated:
        for classes in os.listdir(root_data_path):
            data_list, label_list, moments_list = data_and_labels(os.path.join(root_data_path, classes),
                                                data_list, label_list, moments_list, classes, separation)
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
                        start_list, end_list, separation)
        
    le = preprocessing.LabelEncoder()
    le.fit(label_list)
    label_list = list(le.transform(label_list))
    
    # Save audios, labels and moments in file
    utils.list_to_file(data_list, os.path.join(root_data_path,'data.txt'))
    utils.list_to_file(label_list, os.path.join(root_data_path,'labels.txt'))
    utils.list_to_file(moments_list, os.path.join(root_data_path,'moments.txt'))
    
    return

def classes_combination(root_data_path, equal, combs, speech_pct):
    
    classes = os.listdir(root_data_path)
    speech_path = os.path.join(root_data_path, classes[combs[1]])
    music_path = os.path.join(root_data_path, classes[combs[0]])
    speech_files = [os.path.join(speech_path, i) for i in os.listdir(speech_path)]
    music_files = [os.path.join(music_path, i) for i in os.listdir(music_path)]
    
    for i in range(len(os.listdir(speech_path))):
        print(i)
        speech, sr_speech = librosa.load(speech_files[i])
        music, sr_music = librosa.load(music_files[j])
        speech = librosa.resample(speech, sr_speech, sr_music)
        
        if equal:
            j = j+1
        else:
            while len(music)<len(speech):
                add_music, sr_add_music = librosa.load(music_files[j])
                if not sr_music == sr_add_music:
                    add_music = librosa.resample(add_music, sr_add_music, sr_music)
                music = np.append(music, add_music)
                j = j+1
                if j>=len(music_files):
                    j=0
            music = music[0:len(speech)]
            
        
        comb = speech_pct*speech+(1-speech_pct)*music
        if combs[0]==0:
            folder = 'music_speech'
        elif combs[0]==2:
            folder = 'speech_noise'
        output_path = os.path.join(root_data_path, folder,
                                   'comb_'+str(i)+'.wav')
        wavfile.write(output_path, sr_music, comb)
            
    return

def labels_demo(data_path, file, num_classes):
    labels_old = utils.file_to_list(os.path.join(data_path, file), False)
    if num_classes == 3:
        dct = {'1\n':'2'}
    if num_classes == 4 or num_classes == 5:
        dct = {'1\n':'3'}
    labels = list(map(dct.get, labels_old))
    utils.list_to_file(labels, os.path.join(data_path, 'labels'+str(num_classes)+'.txt'))
    return

