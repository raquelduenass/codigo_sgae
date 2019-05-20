import os
import utils
import numpy as np
import keras
import process_label
from keras import backend as k
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from random import shuffle as sh
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

        # File of database for the phase
        if directory == 'train':
            dirs_file = os.path.join(FLAGS.experiment_root_directory, 'train_files.txt')
            labels_file = os.path.join(FLAGS.experiment_root_directory, 'train_labels.txt')
        elif directory == 'val':
            dirs_file = os.path.join(FLAGS.experiment_root_directory, 'val_files.txt')
            labels_file = os.path.join(FLAGS.experiment_root_directory, 'val_labels.txt')
        else:
            dirs_file = os.path.join(FLAGS.experiment_rootdir, 'test_files.txt')
            labels_file = os.path.join(FLAGS.experiment_root_directory, 'test_labels.txt')
        
        self.file_names, self.ground_truth = cross_val_load(dirs_file, labels_file,
                                                            FLAGS.f_output)
        
        # Number of samples in data
        self.samples = len(self.file_names)
        self.num_classes = len(np.unique(self.ground_truth, axis=0))
        # Check if data is empty
        if self.samples == 0:
            raise IOError("Did not find any data")

        print('Found {} images belonging to {} classes.'.format(
                self.samples, FLAGS.num_classes))

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
            if FLAGS.structure == 'simple':
                x = np.load(self.file_names[j])
                # Data augmentation
                x = self.image_data_generator.standardize(x)
                x = self.image_data_generator.random_transform(x)
            else:
                x = load_many(self, j, FLAGS.wind_len)

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


def cross_val_create(path):
    """

    :param path: folder containing the data
    :return: split of the data in train, validation and test sets
    """
    # File names, moments and labels of all samples in data.
    file_names = utils.file_to_list(os.path.join(path, 'data.txt'))
    labels = utils.file_to_list(os.path.join(path, 'labels.txt'))
    order = list(range(len(file_names)))
    sh(order)
    order = np.asarray(order)
    index4 = int(round(len(order)/4))
    index2 = int(round(len(order)/2))
    
    # Create files of directories, labels and moments
    utils.list_to_file([file_names[i] for i in order[index2:]],
                       os.path.join(FLAGS.experiment_root_directory, 'train_files.txt'))
    utils.list_to_file([file_names[i] for i in order[index4:index2]],
                       os.path.join(FLAGS.experiment_root_directory, 'val_files.txt'))
    utils.list_to_file([file_names[i] for i in order[0:index4]],
                       os.path.join(FLAGS.experiment_root_directory, 'test_files.txt'))
    utils.list_to_file([labels[i] for i in order[index2:]],
                       os.path.join(FLAGS.experiment_root_directory, 'train_labels.txt'))
    utils.list_to_file([labels[i] for i in order[index4:index2]],
                       os.path.join(FLAGS.experiment_root_directory, 'val_labels.txt'))
    utils.list_to_file([labels[i] for i in order[0:index4]],
                       os.path.join(FLAGS.experiment_root_directory, 'test_labels.txt'))
    return


def cross_val_load(dirs_file, labels_file, f_output):
    """

    :param dirs_file: txt file containing the name of the samples
    :param labels_file: txt file containing the labels of each sample
    :param f_output: function of the last layer of the CNN ('sigmoid' or 'softmax')
    :return dirs_list: samples names
    :return labels_list: ground truth
    """
    dirs_list = utils.file_to_list(dirs_file)
    labels_list = utils.file_to_list(labels_file)
    labels_list = process_label.labels_to_number(labels_list, f_output)
    labels_list = [np.array(i) for i in labels_list]
    return dirs_list, labels_list


def load_many(self, j, wind_len):
    """

    :param self:
    :param j:
    :param wind_len:
    :return images:
    """
    a = np.arange(start=(1-wind_len)/2, stop=(wind_len-1)/2+1)
    images = []
    for i in a:
        if j+i < len(self.file_names):
            x = np.load(self.file_names[int(j+i)])
        else:
            x = np.load(self.file_names[int(i)])
        # Data augmentation
        x = self.image_data_generator.standardize(x)
        x = self.image_data_generator.random_transform(x)
        images.append(x)

    return images
