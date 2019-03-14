# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:41:55 2019

@author: rds
"""

from pydub import AudioSegment
import os

music_path = './data/mixed/music'
for music_file in os.listdir(music_path):
    entry_path = os.path.join(music_path, music_file)
    print(entry_path)
    if music_file.endswith('.wav'):
        audio = AudioSegment.from_file(entry_path, format='wav')
    elif music_file.endswith('.mp3'):
        audio = AudioSegment.from_file(entry_path, format='mp3')