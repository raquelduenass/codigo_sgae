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
import process_label
from pydub import AudioSegment
from scipy.io import wavfile
from random import randint
from common_flags import FLAGS
from random import shuffle as sh
import pandas as pd


def data_and_labels(music_path, data_list, label_list, moments_list, label):
    """
    # Arguments:
        music_path: folder containing the data
        data_list:
        label_list:
        moments_list:
        label: ground truth of the subset of samples
    # Return:
        data_list:
        label_list:
        moments_list:
    """
    for music_file in os.listdir(music_path):
        entry_path = os.path.join(music_path, music_file)
        
        if music_file.endswith('.wav'):
            audio = AudioSegment.from_file(entry_path, format="wav")
            for i in range(int(audio.duration_seconds/FLAGS.separation)):
                data_list.append(os.path.abspath(os.path.join(music_path, music_file)))
                moments_list.append(i * FLAGS.separation)
                label_list.append(label)
                
        elif music_file.endswith('.mp3'):
            audio = AudioSegment.from_file(entry_path, format="mp3")
            for i in range(int(audio.duration_seconds/FLAGS.separation)):
                data_list.append(os.path.abspath(os.path.join(music_path, music_file)))
                moments_list.append(i * FLAGS.separation)
                label_list.append(label)
                
    return data_list, label_list, moments_list


def data_and_labels_mu_speak(label, music_file, pre_label, start_list, end_list):
    """
    # Arguments:
        label:
        music_file:
        pre_label:
        start_list:
        end_list:
    # Return:
        label_list:
    """
    moments_list = []
    audio = AudioSegment.from_file(music_file, format="mp3")
    if not(FLAGS.overlap == 0):
        param = FLAGS.separation / FLAGS.overlap
        start_list[:] = [int(x / FLAGS.overlap) for x in start_list]
        end_list[:] = [int(((x / FLAGS.overlap) // param) * param) for x in end_list]
        labels = np.ones((int((int(audio.duration_seconds / FLAGS.overlap)//param) * param),))
        for i in range(int(audio.duration_seconds * param)):
            moments_list.append(i * FLAGS.overlap)
    else:
        start_list[:] = [int(x / FLAGS.separation) for x in start_list]
        end_list[:] = [int(x / FLAGS.separation) for x in end_list]
        labels = np.ones((int(audio.duration_seconds / FLAGS.separation),))
        for i in range(int(audio.duration_seconds / FLAGS.separation)):
            moments_list.append(i * FLAGS.separation)

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


def create_database(root_data_path, separated):
    """
    # Arguments:
        root_data_path: folder containing the data
        separated: boolean indicating if the audio files are separated in
                   folders according to each class (True) or not
    """
    data_list, label_list, moments_list = [], [], []
    
    if separated:
        for classes in os.listdir(root_data_path):
            data_list, label_list, moments_list = data_and_labels(os.path.join(root_data_path, classes),
                                                                  data_list, label_list, moments_list, classes)
        label_list = process_label.labels_to_number(label_list)

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
                
                label_list_m, moments_list_m = data_and_labels_mu_speak('M', audio_file, label_list,
                                                                        start_music, end_music)
                label_list_h = data_and_labels_mu_speak('H', audio_file, label_list_m,
                                                        start_speech, end_speech)
                label_list.append(label_list_h)
                moments_list.append(moments_list_m)
    
    # Save file names, labels and moments in file
    utils.list_to_file(data_list, os.path.join(root_data_path, 'data.txt'))
    utils.list_to_file(label_list, os.path.join(root_data_path, 'labels.txt'))
    utils.list_to_file(moments_list, os.path.join(root_data_path, 'moments.txt'))
    return


def classes_combination(root_data_path, equal, combs, speech_pct):
    """
    Creation of mixed classes files
    # Arguments:
        root_data_path: folder containing the data
        equal: boolean indicating if the files are all of the same length (True)
        combs: array indicating the position of the classes to be mixed
        speech_pct: percentage of the level of speech over another class
    """
    classes = os.listdir(root_data_path)
    speech_path = os.path.join(root_data_path, classes[combs[1]])
    music_path = os.path.join(root_data_path, classes[combs[0]])
    speech_files = [os.path.join(speech_path, i) for i in os.listdir(speech_path)]
    music_files = [os.path.join(music_path, i) for i in os.listdir(music_path)]
    j = 0
    for i in range(len(speech_files)):
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
        else:
            folder = 'speech_noise'
        output_path = os.path.join(root_data_path, folder, 'comb_'+str(i)+'.wav')
        wavfile.write(output_path, sr_music, comb)
    return


def data_files(data_path):
    """
    txt files are created
    # Arguments:
        data_path: folder containing the data
    """
    file_names, labels, moments = [], [], []
    for classes in os.listdir(data_path):
        class_path = os.path.join(data_path, classes)

        if FLAGS.from_audio:
            for files in os.listdir(class_path):
                length = librosa.get_duration(librosa.load(os.path.join(class_path, files))[0])
                for moment in range(0, int(length), FLAGS.separation):
                    moments.append(moment)
                    labels.append(classes)
                    file_names.append(os.path.join(class_path, files))
        else:
            file_names = file_names + [os.path.join(class_path, files) for files in os.listdir(class_path)]
            labels = labels + [classes for files in os.listdir(class_path)]

    utils.list_to_file(file_names, os.path.join(data_path, 'data.txt'))
    utils.list_to_file(labels, os.path.join(data_path, 'labels.txt'))
    utils.list_to_file(moments, os.path.join(data_path, 'moments.txt'))
    return


def get_transitions(duration):
    """
    # Arguments:
        duration:
    # Return:
        transitions:
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
    Creation of randomly mixed audio files
    # Arguments:
        data_path: folder containing the audio files to be mixed
        save_path: folder containing the final mixed audios
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


def cross_val_create(path):
    """

    :param path: folder containing the data
    :return: split of the data in train, validation and test sets
    """
    # File names, moments and labels of all samples in data.
    file_names = utils.file_to_list(os.path.join(path, 'data.txt'))
    labels = utils.file_to_list(os.path.join(path, 'labels.txt'))
    order = list(range(len(file_names)))
    sh(order)
    order = np.asarray(order)
    index4 = int(round(len(order) / 4))
    index2 = int(round(len(order) / 2))

    # Create files of directories, labels and moments
    utils.list_to_file([file_names[i] for i in order[index2:]],
                       os.path.join(FLAGS.experiment_root_directory, 'train_files.txt'))
    utils.list_to_file([file_names[i] for i in order[index4:index2]],
                       os.path.join(FLAGS.experiment_root_directory, 'val_files.txt'))
    utils.list_to_file([file_names[i] for i in order[0:index4]],
                       os.path.join(FLAGS.experiment_root_directory, 'test_files.txt'))
    utils.list_to_file([labels[i] for i in order[index2:]],
                       os.path.join(FLAGS.experiment_root_directory, 'train_labels.txt'))
    utils.list_to_file([labels[i] for i in order[index4:index2]],
                       os.path.join(FLAGS.experiment_root_directory, 'val_labels.txt'))
    utils.list_to_file([labels[i] for i in order[0:index4]],
                       os.path.join(FLAGS.experiment_root_directory, 'test_labels.txt'))

    if FLAGS.from_audio:
        moments = utils.file_to_list(os.path.join(path, 'moments.txt'))
        utils.list_to_file([moments[i] for i in order[index2:]],
                           os.path.join(FLAGS.experiment_root_directory, 'train_moments.txt'))
        utils.list_to_file([moments[i] for i in order[index4:index2]],
                           os.path.join(FLAGS.experiment_root_directory, 'val_moments.txt'))
        utils.list_to_file([moments[i] for i in order[0:index4]],
                           os.path.join(FLAGS.experiment_root_directory, 'test_moments.txt'))
    return


def cross_val_create_df(path):
    """
    :param path: folder containing the data
    :return: split of the data in train, validation and test sets
    """
    val_data = pd.DataFrame()
    test_data = pd.DataFrame()
    train_data = pd.DataFrame()

    for classes in os.listdir(path):

        data = pd.read_csv(os.path.join(path, classes, 'data.csv'))
        audios = data['audio_name'].unique()

        order = list(range(len(audios)))
        sh(order)
        order = np.asarray(order)
        index4 = int(round(len(order) / 4))
        index2 = int(round(len(order) / 2))

        val_data.append(data[data['spec_name'] == audios[index4:index2]])
        test_data.append(data[data['spec_name'] == audios[0:index4]])
        train_data.append(data[data['spec_name'] == audios[index2:]])

    # Create files of directories, labels and moments
    train_data.to_csv(os.path.join(FLAGS.experiment_root_directory, "train.csv"))
    val_data.to_csv(os.path.join(FLAGS.experiment_root_directory, "validation.csv"))
    test_data.to_csv(os.path.join(FLAGS.experiment_root_directory, "test.csv"))

    # if FLAGS.from_audio:
    #     moments = utils.file_to_list(os.path.join(path, 'moments.txt'))
    #     utils.list_to_file([moments[i] for i in order[index2:]],
    #                        os.path.join(FLAGS.experiment_root_directory, 'train_moments.txt'))
    #     utils.list_to_file([moments[i] for i in order[index4:index2]],
    #                        os.path.join(FLAGS.experiment_root_directory, 'val_moments.txt'))
    #     utils.list_to_file([moments[i] for i in order[0:index4]],
    #                        os.path.join(FLAGS.experiment_root_directory, 'test_moments.txt'))
    return


def create_df_database(data_path, save_path):
    """
    Creation of spectrograms from segments of the audio files
    and creation of csv file with association of spectrogram
    files, audio files and the ground truth label
    # Arguments:
        data_path: folder containing the audio files
        save_path: folder to be containing the spectrograms
    """

    for classes in os.listdir(data_path):
        spec_names, labels, audio_names = [], [], []
        j = 0
        class_path = os.path.join(data_path, classes)

        for files in os.listdir(class_path):
            audio_name = os.path.join(class_path, files)
            audio, sr = librosa.load(audio_name)
            audio = librosa.resample(audio, sr, FLAGS.sr)
            length = librosa.get_duration(audio)
            time_set = [np.round(i, decimals=2) for i in list(np.arange(start=0, stop=length, step=FLAGS.separation))]

            for i in range(len(time_set)):
                segment = audio[int(i * FLAGS.separation * FLAGS.sr):
                                int((i + 1) * FLAGS.separation * FLAGS.sr)]
                mel = process_audio.compute_mel_gram(segment)
                spec_name = os.path.join(save_path, classes, 'mel_' + str(j) + '.npy')
                np.save(spec_name, mel)
                j = j + 1

                audio_names.append(audio_name)
                spec_names.append(spec_name)
                labels.append(classes)

        with open(os.path.join(save_path, classes, 'data.csv'), mode='w') as csv_file:
            fieldnames = ['spec_name', 'ground_truth', 'audio_name']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(spec_names)):
                writer.writerow({'spec_name': spec_names[i], 'ground_truth': labels[i], 'audio_name': audio_names[i]})

    return
