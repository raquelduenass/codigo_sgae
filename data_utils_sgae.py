import os
import numpy as np

import keras
from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from pydub import AudioSegment
from common_flags import FLAGS
import utils
from random import shuffle
from pyAudioAnalysis import audioFeatureExtraction

class DataGenerator(ImageDataGenerator):
    """
    Generate minibatches of images and labels with real-time augmentation.
    The only function that changes w.r.t. parent class is the flow that
    generates data. This function needed in fact adaptation for different
    directory structure and labels. All the remaining functions remain
    unchanged.
    """
    # SE REESCRIBE ESTA FUNCION, TODAS LAS DEMAS SE HEREDAN DE IMAGEDATAGENERATOR DE KERAS
    def flow_from_directory(self, directory, num_classes, target_size=(224,224,3),
                            batch_size=32, shuffle=True,
                            seed=None, follow_links=False):
        return DirectoryIterator(
                directory, num_classes, self, target_size=target_size, #img_mode=img_mode,
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

    # TODO: Add functionality to save images to have a look at the augmentation
    """
    def __init__(self, phase, num_classes, image_data_generator,
            target_size=(224,224,3),
            batch_size=32, shuffle=True, seed=None, follow_links=False):
        self.image_data_generator = image_data_generator
        self.target_size = target_size
        self.follow_links = follow_links
        
        # Initialize number of classes
        self.num_classes = num_classes
        
        # Allowed image formats
        self.formats = {'png', 'jpg'}

        # Number of samples in dataset
        self.samples = 0

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
        
        # Number of samples per class
        self.samples_per_class = [self.ground_truth.count(0),
                                  self.ground_truth.count(1)]
        
        # Number of samples in dataset
        self.samples = len(self.filenames) 
        
        # Check if dataset is empty            
        if self.samples == 0:
            raise IOError("Did not find any data")

        # Conversion of list into array
        self.ground_truth = np.array(self.ground_truth, dtype= K.floatx())

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
        current_batch_size = index_array.shape[0]
                    
        # Initialize batch of images
        batch_x = np.zeros((current_batch_size,) + self.target_size,
                dtype=K.floatx())
        # Initialize batch of ground truth
        batch_y = np.zeros((current_batch_size, self.num_classes,),
                                 dtype=K.floatx())

        # Build batch of image data
        for i, j in enumerate(index_array):
            x = create_spectrogram(j, self.moments, self.filenames, self.target_size)
            # Data augmentation
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        # Build batch of labels
        batch_y = np.array(self.ground_truth[index_array], dtype=K.floatx())
        batch_y = keras.utils.to_categorical(batch_y, num_classes=self.num_classes)
        batch_x = np.expand_dims(batch_x, axis=3)
        return batch_x, batch_y

def create_spectrogram(i, moments, data, target_size):
    newAudio = AudioSegment.from_file(data[i], format="wav")
    segment = newAudio[int(moments[i])*1000:(int(moments[i])+1)*1000]

    Fs=20000
    segment = segment.get_array_of_samples()
    pxx, TimeAxis, FreqAxis = audioFeatureExtraction.stSpectogram(segment, Fs, round(Fs * 0.010), round(Fs * 0.020), False)

    return pxx

def cross_val_create(data_path):
    
    # Filenames, moments and labels of all samples in dataset.
    filenames = utils.file_to_list(os.path.join(FLAGS.data_path,'data.txt'))
    moments = utils.file_to_list(os.path.join(FLAGS.data_path,'moments.txt'))
    labels = utils.file_to_list(os.path.join(FLAGS.data_path,'labels.txt'))
        
    order = list(range(len(filenames)))
    shuffle(order)
    order = np.asarray(order)
    index4 = int(round(len(order)/4))
    index2 = int(round(len(order)/2))
    test_files = [filenames[i] for i in order[0:index4]]
    test_moments = [moments[i] for i in order[0:index4]]
    test_labels = [labels[i] for i in order[0:index4]]
    val_files = [filenames[i] for i in order[index4:index2]]
    val_moments = [moments[i] for i in order[index4:index2]]
    val_labels = [labels[i] for i in order[index4:index2]]
    train_files = [filenames[i] for i in order[index2:]]
    train_moments = [moments[i] for i in order[index2:]]
    train_labels = [labels[i] for i in order[index2:]]
    
    # Create files of directories, labels and moments
    utils.list_to_file(train_files, os.path.join(FLAGS.experiment_rootdir, 'train_files.txt'))
    utils.list_to_file(val_files, os.path.join(FLAGS.experiment_rootdir, 'val_files.txt'))
    utils.list_to_file(test_files, os.path.join(FLAGS.experiment_rootdir, 'test_files.txt'))
    utils.list_to_file(train_labels, os.path.join(FLAGS.experiment_rootdir, 'train_labels.txt'))
    utils.list_to_file(val_labels, os.path.join(FLAGS.experiment_rootdir, 'val_labels.txt'))
    utils.list_to_file(test_labels, os.path.join(FLAGS.experiment_rootdir, 'test_labels.txt'))
    utils.list_to_file(train_moments, os.path.join(FLAGS.experiment_rootdir, 'train_moments.txt'))
    utils.list_to_file(val_moments, os.path.join(FLAGS.experiment_rootdir, 'val_moments.txt'))
    utils.list_to_file(test_moments, os.path.join(FLAGS.experiment_rootdir, 'test_moments.txt'))

    return

def cross_val_load(dirs_file, moments_file, labels_file):
    dirs_list = utils.file_to_list(dirs_file)
    labels_list = utils.file_to_list(labels_file)
    moments_list = utils.file_to_list(moments_file)
        
    return dirs_list, moments_list, labels_list
