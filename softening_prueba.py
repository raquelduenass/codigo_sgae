# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 12:04:27 2019

@author: rds
"""
import json
from sklearn import metrics
from scipy.signal import medfilt


def join_labels(prediction, silence):
    labels = ['']*50
    k = 0
    for i in range(len(prediction)):
        while silence[k] != '':
            labels[k] = 'S'
            k = k+1
        labels[k] = prediction[i]
        k = k+1
    return labels


def counting(data, label):
    loc = [i for i in range(len(data)) if data[i] == label]
    if not(len(loc) == 0):
        pos = [loc[0]] + [loc[i+1] for i in range(len(loc)-1) if (not (loc[i]+1 == loc[i+1]))]
        fin = [loc[i] for i in range(len(loc)-1) if (not (loc[i]+1 == loc[i+1]))]
        fin.append(loc[-1])
        length = [fin[i]-pos[i]+1 for i in range(len(pos))]
    else:
        pos = []
        length = []
    return pos, length


def soft_apply(labels, pos, length, th):
    if not(len(pos) == 0):
        for i in range(len(pos)):
            if length[i] < th:
                if not(len(labels) <= pos[i]+length[i]):
                    if labels[pos[i]-1] == labels[pos[i]+length[i]+1]:
                        labels[pos[i]:(pos[i]+length[i])] = [labels[pos[i]-1]]*(length[i])
                    else:
                        add = length[i] % 2
                        labels[pos[i]:(pos[i]+int(length[i]/2))] = [labels[pos[i]-1]]*(int(length[i]/2)+add)
                        labels[(pos[i]+int(length[i]/2)):pos[i]+length[i]] =\
                            [labels[pos[i]+length[i]]]*(int(length[i]/2))
                else:
                    if length[i] > 1:
                        labels[pos[i]:(pos[i]+length[i])] = [labels[pos[i]-1]]*(length[i])
                    else:
                        labels[pos[i]] = labels[pos[i]-1]
    return labels


def softening(labels):
    
    """
    """
    silence_th = 4
    music_th = 5
    non_music_th = 5
    
    # Silence filtering:
    pos, length = counting(labels, 'S')
    labels = soft_apply(labels, pos, length, silence_th)
            
    # Standardization of classes
    for i in range(len(labels)):
        if not (labels[i] == 'M'):
            labels[i] = 'NM'
           
    # Softening non-music class
    pos, length = counting(labels, 'NM')
    labels = soft_apply(labels, pos, length, non_music_th)
    
    # Softening music class
    pos, length = counting(labels, 'M')
    labels = soft_apply(labels, pos, length, music_th)
    
    return labels


with open('./models/test_3/demo_predicted_and_real_labels.json') as f:
    file_data = json.load(f)
predicted_labels = file_data['pred_labels']
real_labels = file_data['real_labels']
separation = 2

# Standardize real labels
for j in range(len(real_labels)):
    if not (real_labels[j] == 'M'):
        real_labels[j] = 'NM'
        
# Accuracy before softening
ave_accuracy = metrics.accuracy_score(real_labels, predicted_labels)
print('Initial accuracy: ', ave_accuracy)

dct = {'MH': 0, 'M': 1, 'H': 2, 'S': 3}
new_dct = {0: 'M', 1: 'M', 2: 'NM', 3: 'NM'}

new_labels = list(map(dct.get, predicted_labels))
softened = medfilt(new_labels, 5)
softened = medfilt(softened, 7)
softened = medfilt(softened, )
softened = list(map(new_dct.get, softened))

# Accuracy after softening
ave_accuracy = metrics.accuracy_score(real_labels, softened)
print('Softening accuracy: ', ave_accuracy)

# Detection of beginning of music
music_pos, music_dur = counting(softened, 'M')
print('Music detected in:')
for j in range(len(music_pos)):
    print('Beginning: ', (music_pos[j]*separation)//60, 'min ', int((music_pos[j]*separation) % 60), 'seg - Duration: ',
          music_dur[j]*separation)
