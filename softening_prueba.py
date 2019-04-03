# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 12:04:27 2019

@author: rds
"""
import json
from sklearn import metrics
import utils
import os
from common_flags import FLAGS


CLASSES = ['M', 'MH', 'N', 'H']

with open('./models/test_5/demo_predicted_and_real_labels.json') as f:
    file_data = json.load(f)
predicted_labels = file_data['predicted_labels']
# soft_labels = file_data['soft_labels']
real_labels = utils.file_to_list(os.path.join(FLAGS.demo_path, 'labels.txt'))
separation = 2
overlap = 0.5
wind_len = 21
param = separation/overlap
index = int((len(real_labels)//param)*param)
real_labels = real_labels[0:index]
real_labels = [CLASSES[int(i)] for i in real_labels]

ave_accuracy = metrics.accuracy_score(real_labels, predicted_labels)
print('Average accuracy before softening= ', ave_accuracy)

soft_labels = utils.soft_max(predicted_labels, wind_len, separation/overlap)
soft_labels = utils.soft_max(soft_labels, wind_len+10, 0)

ave_accuracy = metrics.accuracy_score(real_labels, soft_labels)
print('Average accuracy after softening= ', ave_accuracy)

# Detection of beginning of music
music_pos, music_dur = utils.counting(soft_labels, 'M')

print('Music detected in:')
for j in range(len(music_pos)):
    print('Beginning: ', int((music_pos[j]*overlap)//60), 'min ',
          int((music_pos[j]*overlap) % 60), 'seg - Duration: ',
          music_dur[j]*overlap)
