# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 12:22:16 2019

@author: ich
"""
import numpy as np
import demo_utils
import utils
import os
from common_flags import FLAGS
from keras import backend as K
import gflags
import sys
import librosa

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

def _main():
    
    # Data loading
    data = 'C:/Users/ich/Documents/GitHub/sgae_code/data/muspeak/UTMA-26.wav'
    audio, sr = librosa.load(data)
    
    # Initialization
    segments = []
    spectrograms = []
    CLASSES = ['M','H','R'] # Música, Habla y Ruido
    TEST_PHASE = 1
    
    # Audio segmentation
    length = int(audio.shape[0])
    for i in range(length):
        segments.append(audio[i*sr:(i+1)*sr])
    
    # Create initial label vector
    labels = np.zeros((0,length))
    
    # Silence detection
    for i in range(length):
        if demo_utils.silence_detection(segments[i].all()):
            labels[i]='S'
        else:
            spectrograms.append(demo_utils.compute_melgram(segments[i].all()))
            
    # Class prediction - Classification     
            
    # Set testing mode (dropout/batchnormalization)
    K.set_learning_phase(TEST_PHASE)
    
    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    model = utils.jsonToModel(json_model_path)

    # Load weights
    weights_load_path = os.path.abspath('./models/test_5/weights_002.h5')
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except:
        print("Impossible to find weight path. Returning untrained model")

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Get predictions
    n_samples = len(spectrograms)
    nb_batches = int(np.ceil(n_samples / FLAGS.batch_size))
    probs_per_class = utils.compute_predictions(
            model, spectrograms, nb_batches, verbose = 1)
    
    # Prediced labels
    classes = np.argmax(probs_per_class, axis=-1)
    
    # Join classes - silence labels
    j = 0
    for i in range(len(classes)):
        while labels[j] != 0:
            j = j+1
        labels[j] = CLASSES[classes[i]]
        
    # Class softening
    

def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)
    _main()

if __name__ == "__main__":
    main(sys.argv)
