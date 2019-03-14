import os
import numpy as np

import keras
from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from common_flags import FLAGS
from random import shuffle
import utils
import librosa

class DataGenerator(ImageDataGenerator):
    """
    Generate minibatches of images and labels with real-time augmentation.
    The only function that changes w.r.t. parent class is the flow that
    generates data. This function needed in fact adaptation for different
    directory structure and labels. All the remaining functions remain
    unchanged.
    """
    # SE REESCRIBE ESTA FUNCION, TODAS LAS DEMAS SE HEREDAN DE IMAGEDATAGENERATOR DE KERAS
    def flow_from_directory(self, directory, num_classes, power, sr, separation, target_size=(224,224,3),
                            batch_size=32, shuffle=True,
                            seed=None, follow_links=False):
        return DirectoryIterator(
                directory, num_classes, power, sr, separation, self, target_size=target_size, #img_mode=img_mode,
                batch_size=batch_size, shuffle=shuffle, seed=seed,
                follow_links=follow_links)

class DirectoryIterator(Iterator):
    """
    Class for managing data loading of images and labels

    # Arguments
       phase: training, validation or test stage
       num_classes: Output dimension (number of classes).
       image_data_generator: Image Generator.
       target_size: tuple of integers, dimensions to resize input images to.
       batch_size: The desired batch size
       shuffle: Whether to shuffle data or not
       seed: numpy seed to shuffle data
       follow_links: Bool, whether to follow symbolic links or not
    """
    def __init__(self, phase, num_classes, power, sr, separation, image_data_generator,
            target_size=(224,224,3),
            batch_size=32, shuffle=True, seed=None, follow_links=False):
        
        self.image_data_generator = image_data_generator
        self.target_size = target_size
        self.follow_links = follow_links
        self.num_classes = num_classes
        self.power = power
        self.separation = separation
        self.sr = sr

        # File of database for the phase
        if phase == 'train':
            dirs_file = os.path.join(FLAGS.experiment_rootdir, 'train_files.txt')
            labels_file = os.path.join(FLAGS.experiment_rootdir, 'train_labels.txt')
            moments_file = os.path.join(FLAGS.experiment_rootdir, 'train_moments.txt')
        elif phase == 'val':
            dirs_file = os.path.join(FLAGS.experiment_rootdir, 'val_files.txt')
            labels_file = os.path.join(FLAGS.experiment_rootdir, 'val_labels.txt')
            moments_file = os.path.join(FLAGS.experiment_rootdir, 'val_moments.txt')
        elif phase == 'test':
            dirs_file = os.path.join(FLAGS.experiment_rootdir, 'test_files.txt')
            labels_file = os.path.join(FLAGS.experiment_rootdir, 'test_labels.txt')
            moments_file = os.path.join(FLAGS.experiment_rootdir, 'test_moments.txt')
        
        self.filenames, self.moments, self.ground_truth = cross_val_load(dirs_file, moments_file, labels_file)
        
        # Number of samples in dataset
        self.samples = len(self.filenames) 
        self.segments = separate_audio(self.sr, self.moments, self.filenames)
        self.silence_labels = ['' for x in range(self.samples)]
        
        # Check if dataset is empty            
        if self.samples == 0:
            raise IOError("Did not find any data")

        print('Found {} images belonging to {} classes.'.format(
                self.samples, self.num_classes))

        super(DirectoryIterator, self).__init__(self.samples,
                batch_size, shuffle, seed)

    def next(self):
        """
        Public function to fetch next batch
        # Returns: The next batch of images and commands.
        """
        with self.lock:
            index_array = next(self.index_generator)
            
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        """
        Public function to fetch next batch.
        Image transformation is not under thread lock, so it can be done in
        parallel
        # Returns: The next batch of images and categorical labels.
        """
                    
        # Initialize batches and indexes
        batch_x = []
        indexes = []
        
        # Build batch of image data
        for i, j in enumerate(index_array):
            x = compute_melgram(self.segments[j], self.sr, self.power, self.separation)
            if silence_detection(x):
                self.silence_labels[j] = 'S'
            else:
                # Data augmentation
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x.append(np.resize(x,(100,100)))
                indexes.append(j)

        # Build batch of labels
        batch_y = np.array(self.ground_truth[indexes], dtype=K.floatx())
        batch_y = keras.utils.to_categorical(batch_y, num_classes=self.num_classes)
        batch_x = np.expand_dims(np.asarray(batch_x), axis=3)
    
        return batch_x, batch_y

def cross_val_create(data_path):
    
    # Filenames, moments and labels of all samples in dataset.
    filenames = utils.file_to_list(os.path.join(FLAGS.data_path,'data.txt'), False)
    moments = utils.file_to_list(os.path.join(FLAGS.data_path,'moments.txt'), False)
    labels = utils.file_to_list(os.path.join(FLAGS.data_path,'labels.txt'), False)
        
    order = np.asarray(shuffle(list(range(len(filenames)))))
    index4 = int(round(len(order)/4))
    index2 = int(round(len(order)/2))
    
    # Create files of directories, labels and moments
    utils.list_to_file([filenames[i] for i in order[index2:]],
                       os.path.join(FLAGS.experiment_rootdir, 'train_files.txt'))
    utils.list_to_file([filenames[i] for i in order[index4:index2]],
                       os.path.join(FLAGS.experiment_rootdir, 'val_files.txt'))
    utils.list_to_file([filenames[i] for i in order[0:index4]],
                       os.path.join(FLAGS.experiment_rootdir, 'test_files.txt'))
    utils.list_to_file([labels[i] for i in order[index2:]],
                       os.path.join(FLAGS.experiment_rootdir, 'train_labels.txt'))
    utils.list_to_file([labels[i] for i in order[index4:index2]],
                       os.path.join(FLAGS.experiment_rootdir, 'val_labels.txt'))
    utils.list_to_file([labels[i] for i in order[0:index4]],
                       os.path.join(FLAGS.experiment_rootdir, 'test_labels.txt'))
    utils.list_to_file([moments[i] for i in order[index2:]],
                       os.path.join(FLAGS.experiment_rootdir, 'train_moments.txt'))
    utils.list_to_file([moments[i] for i in order[index4:index2]],
                       os.path.join(FLAGS.experiment_rootdir, 'val_moments.txt'))
    utils.list_to_file([moments[i] for i in order[0:index4]],
                       os.path.join(FLAGS.experiment_rootdir, 'test_moments.txt'))

    return

def cross_val_load(dirs_file, moments_file, labels_file):
    dirs_list = utils.file_to_list(dirs_file, True)
    moments_list = utils.file_to_list(moments_file, True)
    labels_list = utils.file_to_list(labels_file, True)
    labels_list = [int(i.split('\n')[0]) for i in labels_list]
        
    return dirs_list, moments_list, np.array(labels_list, dtype= K.floatx())

def separate_audio(sr, moments, files):
    segments = []
    for j in range(len(files)):
        audio, sr_old = librosa.load(files[j].split('\n')[0])
        audio = librosa.resample(audio, sr_old, sr)
        segments.append(audio[int(moments[j])*sr:(int(moments[j])+1)*sr])
    return segments

def compute_melgram(src, sr, power, dura):
    ''' Compute a mel-spectrogram and returns it in a shape of (96,), where
    96 == #mel-bins and 1366 == #time frame'''

    # mel-spectrogram parameters
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    n_sample = src.shape[0]
    n_sample_fit = int(dura*sr)

    if n_sample < n_sample_fit:  # if too short
        src = np.concatenate([src, np.zeros((int(dura*sr) - n_sample,))])
    elif n_sample > n_sample_fit:  # if too long
        src = src[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]
    mel = librosa.feature.melspectrogram(
            y=src, sr=sr, hop_length=HOP_LEN,
            n_fft=N_FFT, n_mels=N_MELS)
#    elif power == 1:
#        ret = librosa.amplitude_to_db(mel**2)
#    elif power == 2:
    ret = librosa.power_to_db(mel)
    return ret

def silence_detection(audio_slice):
    silence_thresh = -16
    silence = librosa.feature.rmse(audio_slice) <= silence_thresh
    return silence
