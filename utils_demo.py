import os
import numpy as np
import utils
import librosa
import process_audio
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from common_flags import FLAGS


class DataGenerator(ImageDataGenerator):
    """
    Generate mini-batches of images and labels with real-time augmentation.
    The only function that changes w.r.t. parent class is the flow that
    generates data. This function needed in fact adaptation for different
    directory structure and labels. All the remaining functions remain
    unchanged.
    """

    def flow_from_directory(self, num_classes, power, sr, separation, overlap, target_size=(224, 224, 3),
                            batch_size=32, shuffle=False,
                            seed=None, follow_links=False):
        return DirectoryIterator(num_classes, power, sr, separation, overlap, self, target_size=target_size,
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
    def __init__(self, num_classes, power, sr, separation, overlap, image_data_generator,
                 target_size=(224, 224, 3), batch_size=32, shuffle=False, seed=None,
                 follow_links=False):
        
        self.image_data_generator = image_data_generator
        self.target_size = target_size
        self.follow_links = follow_links
        self.power = power
        self.num_classes = num_classes
        self.samples = 0
        self.sr = sr
        self.separation = separation
        self.overlap = overlap

        # File of database for the phase
        self.file_names = utils.file_to_list(os.path.join(FLAGS.demo_path, 'data.txt'))

        # Check if data set is empty
        if len(self.file_names) == 0:
            raise IOError("Did not find any data")

        # Calculate number of samples
        self.files_length = []
        for i in range(len(self.file_names)):
            audio, sr_old = librosa.load(self.file_names[i])
            audio = librosa.resample(audio, sr_old, self.sr)
            if i == 0:
                self.files_length.append(int((librosa.get_duration(audio) -
                                              self.separation) // self.overlap))
            else:
                self.files_length.append(int(self.files_length[i-1]) +
                                         int((librosa.get_duration(audio) -
                                              self.separation) // self.overlap))
        self.samples = self.files_length[len(self.files_length)-1]
        
        # Silence labels
        self.silence_labels = [[]]*self.samples

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

        # Extract segments of all the files
        segments = process_audio.separate_many_audio(self, index_array)

        # Initialize batches and indexes
        batch_x = []
        
        # Build batch of image data
        for i, j in enumerate(index_array):
            x = process_audio.compute_mel_gram(self.separation, self.sr,
                                               self.power, segments[i])
            if process_audio.silence_detection(x):
                self.silence_labels[j] = 'S'
            else:
                self.silence_labels[j] = ''
                # Data augmentation
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x.append(x)

        # Build batch of labels
        batch_x = np.expand_dims(np.asarray(batch_x), axis=3)
    
        return batch_x
