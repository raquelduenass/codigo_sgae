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
    utils.save_to_csv(os.path.join(root_data_path, 'data.csv'), [data_list, label_list, moments_list],
                      ['file_name', 'ground_truth', 'moments'])
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
    txt files are created of separated database
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

    utils.save_to_csv(os.path.join(data_path, 'data.csv'), [file_names, labels, moments],
                      ['file_name', 'ground_truth', 'moments'])
    return


def get_transitions(len_music, len_speech):
    """
    # Arguments:
        duration: total duration of the combination of files
    # Return:
        transitions: moments where a transition is going to take place (in seconds)
    """
    durations_music, durations_speech = [], []
    init_speech = []

    while len(init_speech) == 0:
        n_trans = randint(2, 6)
        for j in range(n_trans):
            durations_music.append(randint(6, int(len_music)))
            durations_speech.append(randint(6, int(len_speech)))

        dur_music = [int(durations_music[i]*len_music/sum(durations_music)) for i in range(n_trans)]
        dur_speech = [int(durations_speech[i]*len_speech/sum(durations_speech)) for i in range(n_trans)]

        durations_music, durations_speech = [], []

        for i in range(n_trans):
            if dur_speech[i] > 5:
                durations_speech.append(dur_speech[i])
            if dur_music[i] > 5:
                durations_music.append(dur_music[i])

        if len(durations_music) > len(durations_speech):
            durations_music = durations_music[:len(durations_speech)]
        else:
            durations_speech = durations_speech[:len(durations_music)]

        init_music = [sum(durations_music[:i+1]) for i in range(len(durations_music))]
        init_speech = [sum(durations_speech[:i+1]) for i in range(len(durations_speech))]

    return init_music, init_speech


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
        seg_music, seg_speech, audio_labels = [], [], []
        case = randint(0, 1)
        music, sr = librosa.load(os.path.join(data_path, classes[0], music_files[i]))
        speech, sr_speech = librosa.load(os.path.join(data_path, classes[1], speech_files[i]))
        speech = librosa.resample(speech, sr_speech, sr)
        label = ['speech', 'music']

        dur_music, dur_speech = get_transitions(librosa.get_duration(music), librosa.get_duration(speech))

        seg_music.append(music[0:dur_music[0] * sr])
        seg_speech.append(speech[0:dur_speech[0] * sr])

        if not(len(dur_music) == 1):
            if len(dur_music) == 2:
                seg_music.append(music[dur_music[0] * sr:(dur_music[1] - 1) * sr])
                seg_speech.append(speech[dur_speech[0] * sr:(dur_speech[1] - 1) * sr])
            else:
                for j in range(len(dur_music)-1):
                    seg_music.append(music[dur_music[j]*sr:(dur_music[j+1]-1)*sr])
                    seg_speech.append(speech[dur_speech[j]*sr:(dur_speech[j+1]-1)*sr])

        segments = [seg_music, seg_speech]
        durations = [dur_music, dur_speech]

        audio = segments[case][0]
        audio_labels.append([label[case]] * (durations[case][0] - 3))

        if not (len(dur_music) == 1):
            if len(dur_music) == 2:
                case = (case + 1) % 2
                dur = durations[case][1] - durations[case][0]
                segment = segments[case][1]

                if len(segment) == 0:
                    break
                else:
                    fade_in, ending = process_audio.fade_in_out(segment, sr)
                    beginning, fade_out = audio[0:len(audio) - 2 * sr - 1], audio[len(audio) - 2 * sr:]
                    audio = np.append(beginning, fade_in + fade_out)
                    audio = np.append(audio, ending)

                    audio_labels = audio_labels + ['music_speech'] * 2
                    audio_labels = audio_labels + [label[case]] * (dur - 5)
            else:
                for j in range(len(dur_music)-1):
                    case = (case + 1) % 2
                    dur = durations[case][j+1] - durations[case][j]
                    segment = segments[case][j+1]

                    if len(segment) == 0:
                        break
                    else:
                        fade_in, ending = process_audio.fade_in_out(segment, sr)
                        beginning, fade_out = audio[0:len(audio)-2*sr-1], audio[len(audio)-2*sr:]
                        audio = np.append(beginning, fade_in+fade_out)
                        audio = np.append(audio, ending)

                        audio_labels = audio_labels + ['music_speech']*2
                        audio_labels = audio_labels + [label[case]] * (dur - 5)

        librosa.feature.rms(audio)
        audio_labels = audio_labels + [label[case]]*2
        audio_labels = audio_labels[0] + audio_labels[1:]
        output_path = os.path.join(save_path, 'comb_'+str(i)+'.wav')
        wavfile.write(output_path, sr, audio)
        file_names.append(output_path)
        labels.append(audio_labels)

    utils.save_to_csv(os.path.join(save_path, "demo.csv"), [file_names, labels], ['file_name', 'ground_truth'])

    return


def cross_val_create_df(path):
    """
    :param path: folder containing the data
    :return: split of the data (depending on the audio file) in train, validation and test sets
    """
    val_gt, test_gt, train_gt = [], [], []
    val_data, test_data, train_data = [], [], []

    for i, classes in enumerate(os.listdir(path)):

        data = pd.read_csv(os.path.join(path, classes, 'data.csv'))
        audios = data['audio_name'].unique()
        order = list(range(len(audios)))
        sh(order)
        order = np.asarray(order)
        index4 = int(round(len(order) / 4))
        index2 = int(round(len(order) / 2))

        val_gt.append(list(data['ground_truth'][data['audio_name'].isin(audios[index4:index2])]))
        test_gt.append(list(data['ground_truth'][data['audio_name'].isin(audios[0:index4])]))
        train_gt.append(list(data['ground_truth'][data['audio_name'].isin(audios[index2:])]))

        val_data.append(list(data['spec_name'][data['audio_name'].isin(audios[index4:index2])]))
        test_data.append(list(data['spec_name'][data['audio_name'].isin(audios[0:index4])]))
        train_data.append(list(data['spec_name'][data['audio_name'].isin(audios[index2:])]))

    val_data = [j for i in val_data for j in i]
    test_data = [j for i in test_data for j in i]
    train_data = [j for i in train_data for j in i]
    val_gt = [j for i in val_gt for j in i]
    test_gt = [j for i in test_gt for j in i]
    train_gt = [j for i in train_gt for j in i]

    fieldnames = ['spec_name', 'ground_truth']
    utils.save_to_csv(os.path.join(FLAGS.experiment_root_directory, "validation.csv"), [val_data, val_gt], fieldnames)
    utils.save_to_csv(os.path.join(FLAGS.experiment_root_directory, "test.csv"), [test_data, test_gt], fieldnames)
    utils.save_to_csv(os.path.join(FLAGS.experiment_root_directory, "train.csv"), [train_data, train_gt], fieldnames)

    return


def create_spectrograms(data_path, save_path):
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

        for files in os.listdir(os.path.join(data_path, classes)):
            audio, sr = librosa.load(os.path.join(os.path.join(data_path, classes), files))
            audio = librosa.resample(audio, sr, FLAGS.sr)
            length = librosa.get_duration(audio)
            time_set = [np.round(i, decimals=2) for i in list(np.arange(start=0, stop=length, step=FLAGS.separation))]

            # for each temporal segment
            for i in range(len(time_set)):
                segment = audio[int(i * FLAGS.separation * FLAGS.sr):
                                int((i + 1) * FLAGS.separation * FLAGS.sr)]
                mel = process_audio.compute_mel_gram(segment, FLAGS.separation)
                spec_name = os.path.join(save_path, classes, 'mel_' + str(j) + '.npy')
                np.save(spec_name, mel)
                j = j + 1

                # Save data for csv file
                audio_names.append(os.path.join(os.path.join(data_path, classes), files))
                spec_names.append(spec_name)
                labels.append(classes)

        utils.save_to_csv(os.path.join(save_path, classes, 'data.csv'), [spec_names, labels, audio_names],
                          ['spec_name', 'ground_truth', 'audio_name'])

    return
