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

def soft_apply(labels, pos, length, th):
    if not(len(pos)==0):
        for i in range(len(pos)):
            if length[i]<th:
                if not(len(labels)<=pos[i]+length[i]):
                    if labels[pos[i]-1]==labels[pos[i]+length[i]+1]:
                        labels[pos[i]:(pos[i]+length[i])] = [labels[pos[i]-1]]*(length[i])
                    else:
                        add = length[i]%2
                        labels[pos[i]:(pos[i]+int(length[i]/2))] = [labels[pos[i]-1]]*(int(length[i]/2)+add)
                        labels[(pos[i]+int(length[i]/2)):pos[i]+length[i]] = [labels[pos[i]+length[i]]]*(int(length[i]/2))
                else:
                    if length[i]>1:
                        labels[pos[i]:(pos[i]+length[i])] = [labels[pos[i]-1]]*(length[i])
                    else:
                        labels[pos[i]] = labels[pos[i]-1]
    return labels

def softening(pred_labels):
    
    """
    """
    labels = pred_labels
    silence_th = 4
    music_th = 20
    non_music_th = 20
    
    # Silence filtering:
    silence_pos, silence_len = counting(labels, 'S')
    labels = soft_apply(labels, silence_pos, silence_len, silence_th)
            
    # Standarization of classes
    for i in range(len(labels)):
        if not (labels[i]== 'M'):
            labels[i]='NM'
           
    # Softening non-music class
    non_music_pos, non_music_len = counting(labels,'NM')
    labels = soft_apply(labels, non_music_pos, non_music_len, non_music_th)
    
    # Softening music class
    music_pos, music_len = counting(labels,'M')
    labels = soft_apply(labels, music_pos, music_len, music_th)
    
    return labels

import json
from sklearn import metrics
from scipy.signal import medfilt

with open('./models/test_3/demo_predicted_and_real_labels.json') as f:
    data = json.load(f)
pred_labels = data['pred_labels']
real_labels = data['real_labels']
separacion = 2

# Standardize real labels
for i in range(len(real_labels)):
    if not (real_labels[i]== 'M'):
        real_labels[i]='NM'
        
# Accuracy before softening
ave_accuracy = metrics.accuracy_score(real_labels,pred_labels)
print('Initial accuracy: ', ave_accuracy)

#labels = pred_labels
dct = {'MH':0,'M':1,'H':2,'S':3}
new_dct = {0:'M',1:'M',2:'NM', 3:'NM'}
labels = list(map(dct.get, pred_labels))
softened = medfilt(labels, 9)
softened = medfilt(softened, 15)
#softened = medfilt(softened, 21)
softened = list(map(new_dct.get, softened))

# Accuracy after softening
ave_accuracy = metrics.accuracy_score(real_labels,softened)
print('Softening accuracy: ', ave_accuracy)

# Detection of beginning of music
music_pos, music_dur = counting(softened, 'M')
print('Music detected in:')
for i in range(len(music_pos)):
    print('Inicio: ',(music_pos[i]*separacion)//60, 'min ', int((music_pos[i]*separacion)%60), 'seg - Duración: ', music_dur[i]*separacion)
    