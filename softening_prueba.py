# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 12:04:27 2019

@author: rds
"""
import json
import utils
import os
from common_flags import FLAGS
import label_process


CLASSES = ['M', 'MH', 'N', 'H']
num_classes = 4
sr = 22050
separation = 2
power = 2
overlap = 0.5
wind_len = 9

with open('./models/test_5/demo_predicted_labels.json') as f:
    file_data = json.load(f)
predicted_labels = file_data['predicted_labels']
files_length = file_data['files_length']
predicted_labels = label_process.separate_labels(predicted_labels, files_length)
real = utils.file_to_list(os.path.join(FLAGS.demo_path, 'labels.txt'))
real_labels = [[]] * 3
for j in range(len(files_length)):
    real[j] = ((real[j].split('[')[1]).split(']')[0]).split(', ')
    real_labels[j] = [CLASSES[int(i)] for i in real[j]]
# probs_over = label_process.separate_labels(prob_per_class, files_length)

# Class softening
soft_labels = label_process.soft_max(predicted_labels, wind_len, separation / overlap, 3)
# soft_labels = label_process.soft_max_prob(predicted_labels, wind_len, separation/overlap,
#                                   len(test_generator.files_length), probs_over)

# Save predicted and softened labels as a dictionary
labels_dict = {'predicted_labels': predicted_labels,
               'soft_labels': soft_labels}
utils.write_to_file(labels_dict, os.path.join(FLAGS.experiment_rootdir,
                                              'demo_predicted_and_soft_labels.json'))

label_process.show_metrics(real_labels, predicted_labels, soft_labels)  # , probs_over)

# Detection of beginning of music
for j in range(len(files_length)):
    music_pos, music_dur = label_process.counting(soft_labels[j], 'M')
    print('Music detected in:')
    for i in range(len(music_pos)):
        if overlap == 0:
            print('Beginning: ', (music_pos[i] * separation) // 60, 'min ',
                  int((music_pos[i] * separation) % 60),
                  'seg - Duration: ', music_dur[i] * separation)
        else:
            print('Beginning: ', int((music_pos[i] * overlap) // 60), 'min ',
                  int((music_pos[i] * overlap) % 60),
                  'seg - Duration: ', music_dur[i] * overlap)
