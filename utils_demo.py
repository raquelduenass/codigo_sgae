import os
import numpy as np
import utils
import librosa
import process_audio
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

    def flow_from_directory(self, batch_size=32, shuffle=False,
                            seed=None, follow_links=False):
        return DirectoryIterator(self, batch_size=batch_size, shuffle=shuffle,
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
    def __init__(self, image_data_generator, batch_size=32, shuffle=False, seed=None,
                 follow_links=False):
        
        self.image_data_generator = image_data_generator
        self.follow_links = follow_links
        self.samples = 0

        # File of database for the phase
        data = pd.read_csv(os.path.join(FLAGS.demo_path, 'data.csv'))
        self.file_names = data['file_name'].unique()

        # Check if data set is empty
        if len(self.file_names) == 0:
            raise IOError("Did not find any data")

        # Calculate number of samples
        self.files_length = []
        for i in range(len(self.file_names)):
            audio, sr_old = librosa.load(self.file_names[i])
            audio = librosa.resample(audio, sr_old, FLAGS.sr)
            if i == 0:
                if FLAGS.overlap == 0:
                    self.files_length.append(int(librosa.get_duration(audio) // FLAGS.separation))
                else:
                    self.files_length.append(int((librosa.get_duration(audio) -
                                                  FLAGS.separation) // FLAGS.overlap))
            else:
                if FLAGS.overlap == 0:
                    self.files_length.append(int(self.files_length[i - 1]) +
                                             int(librosa.get_duration(audio) //
                                                 FLAGS.separation))
                else:
                    self.files_length.append(int(self.files_length[i-1]) +
                                             int((librosa.get_duration(audio) -
                                                  FLAGS.separation) // FLAGS.overlap))
        self.samples = self.files_length[-1]

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
        # Extract segments of all the files
        if FLAGS.structure == 'simple':
            segments = process_audio.separate_many_audio(self, index_array)
        else:
            segments = [[]]*FLAGS.wind_len
            a = np.arange(start=(1 - FLAGS.wind_len) / 2, stop=(FLAGS.wind_len - 1) / 2 + 1)
            for i, j in enumerate(a):
                segments[i] = process_audio.separate_many_audio(self, index_array+j)

        # Initialize batches and indexes
        batch_x, batch_x_wind = [], [[]]*FLAGS.wind_len
        
        # Build batch of image data
        for i, j in enumerate(index_array):
            if FLAGS.structure == 'simple':
                x = process_audio.compute_mel_gram(segments[i], FLAGS.separation)
                # Data augmentation
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)

            else:
                x = load_many(self, segments, i)
            batch_x.append(x)

        # Build batch of labels
        if FLAGS.structure == 'complex':

            for i in range(FLAGS.wind_len):
                batch_x_wind[i] = [np.expand_dims(np.array(batch_x[j][i]), axis=3)
                                   for j in range(len(index_array))]

            batch_x = np.expand_dims(np.array(batch_x), axis=4)

            for i in range(len(index_array)):
                batch_x[i] = [batch_x_wind[j][i] for j in range(FLAGS.wind_len)]
        else:
            batch_x = np.expand_dims(np.array(batch_x), axis=3)
    
        return batch_x


def load_many(self, segments, j):
    images = []
    for i in range(FLAGS.wind_len):
        x = process_audio.compute_mel_gram(segments[i][j], FLAGS.separation)

        # Data augmentation
        x = self.image_data_generator.standardize(x)
        x = self.image_data_generator.random_transform(x)
        images.append(x)
    return images
