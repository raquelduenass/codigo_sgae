# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:41:55 2019
@author: rds
"""

import database_process
music_path = 'C:/Users/rds/Documents/GitHub/data_sgae/mixed'
spec_path = 'C:/Users/rds/Documents/GitHub/data_sgae/spectrograms_npy'

database_process.extract_spec_grams(music_path, spec_path)
