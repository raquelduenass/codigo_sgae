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
    pos = [loc[i] for i in range(len(loc)) if (not (data[loc[i]] == data[loc[i]-1])) or loc[i]==0]
    fin = [loc[i] for i in range(len(loc)) if (not (data[loc[i]] == data[loc[i]+1]))]
    if loc[-1] == len(data)-1:
        fin.append(len(data)-1)
    length = [fin[i]-pos[i]+1 for i in range(len(pos))]
    return pos, length

def softening(labels):
    """
    """
    silence_th = 4
    music_th = 2
    non_music_th = 2
    
    # Silence filtering:
    silence_pos, silence_len = counting(labels, 'S')
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
    for i in range(len(music_pos)):
        if music_len[i]<music_th:
            if labels[music_pos[i]-1]==labels[music_pos[i]+music_len[i]+1]:
                labels[music_pos[i]:music_pos[i]+music_len[i]] = [labels[music_pos[i]-1]]*(music_len[i])
           
    # Softening non-music class
    non_music_pos, non_music_len = counting(labels,'NM')
    for i in range(len(non_music_pos)):
        if non_music_len[i]<non_music_th:
            if labels[non_music_pos[i]-1]==labels[non_music_pos[i]+non_music_len[i]+1]:
                labels[non_music_pos[i]:non_music_pos[i]+non_music_len[i]] = [labels[non_music_pos[i]-1]]*(non_music_len[i])
                
    return labels

silence_labels = ['' for x in range(50)]
pred = ['NM' for x in range(46)]
pred[2] = 'M'
silence_labels[1] = 'S'
silence_labels[2] = 'S'
silence_labels[6] = 'S'
silence_labels[7] = 'S'
labels = join_labels(pred, silence_labels)
soft_labels = softening(labels)
