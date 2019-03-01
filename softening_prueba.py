# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 12:04:27 2019

@author: rds
"""
def join_labels(pred, silence):
    labels = ['' for x in range(50)]
    j = 0
    for i in range(len(pred)):
        while silence[j] != '':
            labels[j] = 'S'
            j = j+1
        labels[j] = pred[i]
        j = j+1
    return labels

def counting(data, label):
    loc = [i for i in range(len(data)) if data[i]==label]
    if not(len(loc)==0):
        pos = [loc[0]] + [loc[i+1] for i in range(len(loc)-1) if (not (loc[i]+1 == loc[i+1]))]
        fin = [loc[i] for i in range(len(loc)-1) if (not (loc[i]+1 == loc[i+1]))]
        fin.append(loc[-1])
        length = [fin[i]-pos[i]+1 for i in range(len(pos))]
    else:
        pos = []
        length = []
    return pos, length

def softening(pred_labels):
    
    """
    """
    labels = pred_labels
    silence_th = 4
    music_th = 5
    non_music_th = 20
    
    # Silence filtering:
    silence_pos, silence_len = counting(labels, 'S')
    if not(len(silence_pos)==0):
        for i in range(len(silence_pos)):
            if silence_len[i]<silence_th:
                if labels[silence_pos[i]-1]==labels[silence_pos[i]+silence_len[i]]:
                    labels[silence_pos[i]:silence_pos[i]+silence_len[i]] = [labels[silence_pos[i]-1]]*(silence_len[i])
                else:
                    labels[silence_pos[i]:silence_pos[i]+int(silence_len[i]/2)] = [labels[silence_pos[i]-1]*(int(silence_len[i]/2))]
                    labels[silence_pos[i]+int(silence_len[i]/2):silence_pos[i]+silence_len[i]] = [labels[silence_pos[i]+silence_len[i]]*(int(silence_len[i]/2))]
                    #   IDEAL: RECLASIFICACIÓN POR CNN
               
#    # Standarization of classes
    for i in range(len(labels)):
        if not (labels[i]== 'M'):
            labels[i]='NM'
            
           
    # Softening music class
    music_pos, music_len = counting(labels,'M')
    if not(len(music_pos)==0):
        for i in range(len(music_pos)):
            if music_len[i]<music_th:
                if labels[music_pos[i]-1]==labels[music_pos[i]+music_len[i]+1]:
                    labels[music_pos[i]:music_pos[i]+music_len[i]] = [labels[music_pos[i]-1]]*(music_len[i])
           
    # Softening non-music class
    non_music_pos, non_music_len = counting(labels,'NM')
    if not(len(non_music_pos)==0):
        for i in range(len(non_music_pos)):
            if non_music_len[i]<non_music_th:
                if labels[non_music_pos[i]-1]==labels[non_music_pos[i]+non_music_len[i]+1]:
                    labels[non_music_pos[i]:non_music_pos[i]+non_music_len[i]] = [labels[non_music_pos[i]-1]]*(non_music_len[i])
                
    return labels

import json
from sklearn import metrics

with open('./models/test_1/demo_predicted_and_real_labels.json') as f:
    data = json.load(f)
pred_labels = data['pred_labels']
soft_labels = data['soft_labels']
real_labels = data['real_labels']

# Standardize real labels
for i in range(len(real_labels)):
    if not (real_labels[i]== 'M'):
        real_labels[i]='NM'
        
# Accuracy before softening
ave_accuracy = metrics.accuracy_score(real_labels,pred_labels)
print('Initial accuracy: ', ave_accuracy)

labels = pred_labels
softened = softening(labels)
for i in range(4):
    softened = softening(softened)

# Accuracy after softening
ave_accuracy = metrics.accuracy_score(real_labels,softened)
print('Softening accuracy: ', ave_accuracy)
    