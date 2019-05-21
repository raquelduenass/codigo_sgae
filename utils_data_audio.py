import os
import utils
import numpy as np
import keras
import librosa
import process_audio
import process_label
from keras import backend as k
from keras.preprocessing.image import Iterator, ImageDataGenerator
from common_flags import FLAGS


class DataGenerator(ImageDataGenerator):
    """
    Generate mini-batches of images and labels with real-time augmentation.
    The only function that changes w.r.t. parent class is the flow that
    generates data. This function needed in fact adaptation for different
    directory structure and labels. All the remaining functions remain
    unchanged.
    """

    def flow_from_directory(self, directory, target_size=(96, 173), color_mode='rgb',
                            classes=None, class_mode='categorical', batch_size=32, shuffle=True,
                            seed=None, save_to_dir=False, save_prefix='', save_format='png',
                            follow_links=False, subset=None, interpolation='nearest'):

        return DirectoryIterator(directory, self, batch_size=batch_size, shuffle=shuffle,
                                 seed=seed, follow_links=follow_links)


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
    def __init__(self, directory, image_data_generator,
                 batch_size=32, shuffle=True, seed=None, follow_links=False):
        
        self.image_data_generator = image_data_generator
        self.follow_links = follow_links

        # File of database for the phase
        if directory == 'train':
            dirs_file = os.path.join(FLAGS.experiment_root_directory, 'train_files.txt')
            moments_file = os.path.join(FLAGS.experiment_root_directory, 'train_moments.txt')
            labels_file = os.path.join(FLAGS.experiment_root_directory, 'train_labels.txt')
        elif directory == 'val':
            dirs_file = os.path.join(FLAGS.experiment_root_directory, 'val_files.txt')
            moments_file = os.path.join(FLAGS.experiment_root_directory, 'val_moments.txt')
            labels_file = os.path.join(FLAGS.experiment_root_directory, 'val_labels.txt')
        else:
            dirs_file = os.path.join(FLAGS.experiment_root_directory, 'test_files.txt')
            moments_file = os.path.join(FLAGS.experiment_root_directory, 'test_moments.txt')
            labels_file = os.path.join(FLAGS.experiment_root_directory, 'test_labels.txt')
        
        self.file_names, self.moments, self.ground_truth = cross_val_load(dirs_file, labels_file, moments_file)
        
        # Number of samples in data
        self.samples = len(self.file_names)
        self.num_classes = len(set(self.ground_truth))
        # Check if data is empty
        if self.samples == 0:
            raise IOError("Did not find any data")

        print('Found {} images belonging to {} classes.'.format(
                self.samples, self.num_classes))

        super(DirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)

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

        # Initialize batches
        batch_x, batch_y_p, batch_x_wind = [], [], [[]]*FLAGS.wind_len

        # Build batch of image data
        for i, j in enumerate(index_array):
            segment, sr = librosa.load(self.file_names[j], offset=self.moments[j], duration=FLAGS.separation)
            x = process_audio.compute_mel_gram(segment)
            # Data augmentation
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x.append(x)
            batch_y_p.append(self.ground_truth[j])

        # Build batch of labels
        if FLAGS.f_output == 'softmax':
            batch_y = np.array(batch_y_p, dtype=k.floatx())
            batch_y = keras.utils.to_categorical(batch_y, num_classes=FLAGS.num_classes)
        else:
            batch_y = batch_y_p

        for i in range(FLAGS.wind_len):
            batch_x_wind[i] = [np.expand_dims(np.array(batch_x[j][i]), axis=3)
                               for j in range(FLAGS.batch_size)]

        batch_x = np.expand_dims(np.array(batch_x), axis=4)

        for i in range(len(index_array)):
            batch_x[i] = [batch_x_wind[0][i], batch_x_wind[1][i], batch_x_wind[2][i],
                          batch_x_wind[3][i], batch_x_wind[4][i]]

        return batch_x, np.asarray(batch_y)


def cross_val_load(dirs_file, labels_file, moments_file=None):
    """
    # Arguments:
        dirs_file: txt file containing the name of the samples
        moments_file:
        labels_file: txt file containing the labels of each sample
    # Return:
        dirs_list: samples names
        moments_list:
        labels_list: ground truth
    """
    dirs_list = utils.file_to_list(dirs_file)
    labels_list = utils.file_to_list(labels_file)
    labels_list = process_label.labels_to_number(labels_list)
    labels_list = [np.array(i) for i in labels_list]

    if FLAGS.from_audio:
        moments_list = utils.file_to_list(moments_file)
        moments_list = [int(i) for i in moments_list]
        return dirs_list, np.array(moments_list, dtype=k.floatx()), labels_list
        # np.array(labels_list, dtype=k.floatx())
    else:
        return dirs_list, labels_list
