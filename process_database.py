# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:35:32 2018

@author: rds
"""
from __future__ import print_function
import os
import csv
import utils
import librosa
import numpy as np
import process_audio
from pydub import AudioSegment
from scipy.io import wavfile
from random import randint


def data_and_labels(music_path, data_list, label_list, moments_list, label, separation):
    """

    :param music_path:
    :param data_list:
    :param label_list:
    :param moments_list:
    :param label:
    :param separation:
    :return:
    """
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


def data_and_labels_mu_speak(label, music_file, pre_label,
                             start_list, end_list, separation, overlap):
    """

    :param label:
    :param music_file:
    :param pre_label:
    :param start_list:
    :param end_list:
    :param separation:
    :param overlap:
    :return:
    """
    moments_list = []
    audio = AudioSegment.from_file(music_file, format="mp3")
    if not(overlap == 0):
        param = separation / overlap
        start_list[:] = [int(x/overlap) for x in start_list]
        end_list[:] = [int(((x/overlap)//param)*param) for x in end_list]
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
        return label_list, moments_list

    else:
        for i in range(len(start_list)):
            for j in range(start_list[i], end_list[i]):
                if pre_label[j] == 0:
                    pre_label[j] = 1
                else:
                    pre_label[j] = 2
        label_list = list(pre_label)
        return label_list


def create_database(root_data_path, separated, separation, overlap):
    """

    :param root_data_path:
    :param separated:
    :param separation:
    :param overlap:
    :return:
    """
    data_list, label_list, moments_list = [], [], []
    
    if separated:
        for classes in os.listdir(root_data_path):
            data_list, label_list, moments_list = data_and_labels(os.path.join(root_data_path, classes),
                                                                  data_list, label_list, moments_list,
                                                                  classes, separation)
        label_list = utils.labels_to_number(label_list)

    else:
        for csv_file in os.listdir(os.path.join(root_data_path, 'meta')):

            start_music, end_music, start_speech, end_speech = [], [], [], []

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
                
                label_list_m, moments_list_m = data_and_labels_mu_speak(
                    'M', audio_file, label_list,
                    start_music, end_music, separation, overlap)
                label_list_h = data_and_labels_mu_speak(
                    'H', audio_file, label_list_m,
                    start_speech, end_speech, separation, overlap)
                label_list.append(label_list_h)
                moments_list.append(moments_list_m)
    
    # Save audios, labels and moments in file
    utils.list_to_file(data_list, os.path.join(root_data_path, 'data.txt'))
    utils.list_to_file(label_list, os.path.join(root_data_path, 'labels.txt'))
    utils.list_to_file(moments_list, os.path.join(root_data_path, 'moments.txt'))
    return


def classes_combination(root_data_path, equal, combs, speech_pct):
    """

    :param root_data_path:
    :param equal:
    :param combs:
    :param speech_pct:
    :return:
    """
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


def extract_spec_grams(data_path, save_path):
    """

    :param data_path:
    :param save_path:
    :return:
    """
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
                    mel = process_audio.compute_mel_gram(separation, sr, power, segment)
                    np.save(os.path.join(save_path, classes, 'mel_'+str(j)+'.npy'), mel)
                    j = j+1


def data_files(data_path, separation):
    """

    :param data_path:
    :param separation:
    :return:
    """
    file_names, labels, moments = [], [], []
    for classes in os.listdir(data_path):
        class_path = os.path.join(data_path, classes)
        for files in os.listdir(class_path):
            length = librosa.get_duration(librosa.load(os.path.join(class_path, files))[0])
            for moment in range(0, int(length), separation):
                moments.append(moment)
                labels.append(classes)
                file_names.append(os.path.join(class_path, files))

    utils.list_to_file(file_names, os.path.join(data_path, 'data.txt'))
    utils.list_to_file(labels, os.path.join(data_path, 'labels.txt'))
    utils.list_to_file(moments, os.path.join(data_path, 'moments.txt'))
    return


def get_transitions(duration):
    """

    :param duration:
    :return:
    """
    init_transitions, transitions = [], []
    for j in range(randint(1, 9)):
        init_transitions.append(randint(0, int(duration)))
    init_transitions.sort()
    if init_transitions[0] >= 2:
        transitions.append(init_transitions[0])
    for k in range(len(init_transitions) - 1):
        if init_transitions[k + 1] - init_transitions[k] > 4:
            transitions.append(init_transitions[k + 1])
    transitions.sort()
    return transitions


def create_manual_demo(data_path, save_path):
    """

    :param data_path:
    :param save_path:
    :return:
    """
    classes = os.listdir(data_path)
    music_files = os.listdir(os.path.join(data_path, classes[0]))
    speech_files = os.listdir(os.path.join(data_path, classes[1]))
    file_names, labels = [], []

    for i in range(len(music_files)):
        audio_labels = []
        music, sr_music = librosa.load(os.path.join(data_path, classes[0], music_files[i]))
        speech, sr_speech = librosa.load(os.path.join(data_path, classes[1], speech_files[i]))
        speech = librosa.resample(speech, sr_speech, sr_music)
        duration = librosa.get_duration(music) + librosa.get_duration(speech)
        gender = [music, speech]
        initial = randint(0, 1)
        last = [0, 0]
        transitions = get_transitions(duration)

        comb = gender[initial][0:transitions[0] * sr_speech]
        if initial == 0:
            audio_labels = audio_labels + ['music']*(transitions[0]-2)
        else:
            audio_labels = audio_labels + ['speech']*(transitions[0]-2)
        last[initial] = transitions[0]
        for j in range(len(transitions)-1):
            case = (j + 1) % 2
            dur = transitions[j+1] - transitions[j]
            segment = gender[case][last[case]*sr_music:(last[case]+dur)*sr_music]
            if len(segment) == 0:
                break
            else:
                fade_in, ending = process_audio.fade_in_out(segment, sr_music)
                beginning, fade_out = comb[0:len(comb)-2*sr_music-1], comb[len(comb)-2*sr_music:]
                comb = np.append(beginning, fade_in+fade_out)
                comb = np.append(comb, ending)
                audio_labels = audio_labels + ['music_speech']*2
                if case == 0:
                    audio_labels = audio_labels + ['music']*(dur-2)
                else:
                    audio_labels = audio_labels + ['speech']*(dur-2)
        output_path = os.path.join(save_path, 'comb_'+str(i)+'.wav')
        wavfile.write(output_path, sr_music, comb)
        file_names.append(output_path)
        labels.append(audio_labels)
    utils.list_to_file(file_names, os.path.join(save_path, 'data.txt'))
    utils.list_to_file(labels, os.path.join(save_path, 'labels.txt'))
    return


data_files('C:/Users/rds/Documents/GitHub/data_sgae/mixed', 2)
