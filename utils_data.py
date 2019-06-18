import os
import numpy as np
import process_label
import pandas as pd
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
        return DirectoryIterator(directory, self, target_size=target_size,
                                 batch_size=batch_size, shuffle=shuffle, seed=seed, follow_links=follow_links)


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
                 target_size=(96, 173),
                 batch_size=32, shuffle=True, seed=None, follow_links=False):
        
        self.image_data_generator = image_data_generator
        self.target_size = target_size
        self.follow_links = follow_links
        self.num_classes = 0

        # File of database for the phase
        if directory == 'train':
            data_file = os.path.join(FLAGS.experiment_root_directory, 'train.csv')
        elif directory == 'val':
            data_file = os.path.join(FLAGS.experiment_root_directory, 'validation.csv')
        else:
            data_file = os.path.join(FLAGS.experiment_root_directory, 'test.csv')

        self.file_names, self.ground_truth = cross_val_load_df(data_file, self)
        
        # Number of samples in data
        self.samples = len(self.file_names)

        # Check if data is empty
        if self.samples == 0:
            raise IOError("Did not find any data")

        print('Found {} images belonging to {} classes.'.format(self.samples, FLAGS.num_classes))

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
        batch_x, batch_y, batch_x_wind = [], [], [[]]*FLAGS.wind_len
        
        # Build batch of image data and labels
        for i, j in enumerate(index_array):
            x = load_many(self, j)
            batch_x.append(x)
            batch_y.append(self.ground_truth[j])

        for i in range(FLAGS.wind_len):
            batch_x_wind[i] = [np.expand_dims(np.array(batch_x[j][i]), axis=3)
                               for j in range(len(batch_x))]

        batch_x = np.expand_dims(np.array(batch_x), axis=4)

        for i in range(len(index_array)):
            batch_x[i] = [batch_x_wind[j][i] for j in range(FLAGS.wind_len)]

        return batch_x, np.asarray(batch_y)


def cross_val_load_df(data_file, self):
    """
    # Arguments:
        data_file: csv file containing the name and ground truth of the samples
    # Return:
        dirs_list: samples names
        moments_list: temporal initiation of the classifying segments
        labels_list: ground truth
    """
    data = pd.read_csv(data_file)
    dirs_list = data['spec_name'].tolist()
    labels_list = process_label.labels_to_number(data['ground_truth'].tolist())
    self.num_classes = len(data['ground_truth'].unique())
    labels_list = [np.array(i) for i in labels_list]

    return dirs_list, labels_list


def load_many(self, j):
    """
    Returns a list of (wind_len) spectrograms, being the one in the middle,
    the one to be classified and the others, the temporally adjacent ones
    """
    a = np.arange(start=(1-FLAGS.wind_len)/2, stop=(FLAGS.wind_len-1)/2+1)
    images = []
    for i in a:
        if j + i < 0:
            # x = np.load(FLAGS.data_path + self.file_names[int(len(self.file_names) + i)])
            x = np.load(self.file_names[int(len(self.file_names) + i)])
        elif j+i < len(self.file_names):
            # x = np.load(FLAGS.data_path + self.file_names[int(j + i)])
            x = np.load(self.file_names[int(j+i)])
        else:
            # x = np.load(FLAGS.data_path + self.file_names[int(i)])
            x = np.load(self.file_names[int(i)])
        # Data augmentation
        x = self.image_data_generator.standardize(x)
        x = self.image_data_generator.random_transform(x)
        images.append(x)

    return images
