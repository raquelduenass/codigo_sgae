import gflags
import numpy as np
import os
import sys
import utils
import utils_demo
import process_label
from keras import backend as k
from common_flags import FLAGS

# Constants
TEST_PHASE = 1
CLASSES = ['M', 'MH', 'H', 'R']
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
os.environ["PATH"] += os.pathsep + 'C:/Users/rds/Downloads/ffmpeg/bin'


def _main():

    # Set testing mode (dropout/batch normalization)
    k.set_learning_phase(TEST_PHASE)
    
    # Output dimension
    num_classes = 4
    sr = 22050
    separation = 2
    power = 2
    overlap = 0.5
    wind_len = 9

    # Generate testing data
    test_data_gen = utils_demo.DataGenerator(rescale=1./255)
    
    # Iterator object containing testing data to be generated batch by batch
    test_generator = test_data_gen.flow_from_directory(num_classes,
                                                       power, sr,
                                                       separation,
                                                       overlap,
                                                       shuffle=False,
                                                       target_size=(FLAGS.img_height, FLAGS.img_width),
                                                       batch_size=FLAGS.batch_size)

    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    model = utils.json_to_model(json_model_path)

    # Load weights
    weights_load_path = os.path.abspath('./models/test_5/weights_010.h5')
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except ImportError:
        print("Impossible to find weight path. Returning untrained model")

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    # Get predictions and ground truth
    n_samples = test_generator.samples
    nb_batches = int(np.ceil(n_samples / FLAGS.batch_size))
    prob_per_class = utils.compute_predictions(
            model, test_generator, nb_batches, verbose=1)
    
    # Predicted labels
    silence_labels = test_generator.silence_labels
    classes = [CLASSES[i] for i in np.argmax(prob_per_class, axis=-1)]
    predicted_labels = process_label.join_labels(classes, silence_labels)

    # Save for test
    labels_dict = {'predicted_labels': predicted_labels,
                   'files_length': test_generator.files_length}
    utils.write_to_file(labels_dict, os.path.join(FLAGS.experiment_rootdir,
                                                  'demo_predicted_labels.json'))
    predicted_labels = process_label.separate_labels(predicted_labels, test_generator.files_length)
    real = utils.file_to_list(os.path.join(FLAGS.demo_path, 'labels.txt'))
    real_labels = [[]] * 3
    for j in range(len(test_generator.files_length)):
        real[j] = ((real[j].split('[')[1]).split(']')[0]).split(', ')
        real_labels[j] = [CLASSES[int(i)] for i in real[j]]
    # probs_over = label_process.separate_labels(prob_per_class, files_length)

    # Class softening
    soft_labels = process_label.soft_max(predicted_labels, wind_len, separation / overlap, 3)
    # soft_labels = label_process.soft_max_prob(predicted_labels, wind_len, separation/overlap,
    #                                   len(test_generator.files_length), probs_over)

    # Save predicted and softened labels as a dictionary
    labels_dict = {'predicted_labels': predicted_labels,
                   'soft_labels': soft_labels}
    utils.write_to_file(labels_dict, os.path.join(FLAGS.experiment_rootdir,
                                                  'demo_predicted_and_soft_labels.json'))

    process_label.show_metrics(real_labels, predicted_labels, soft_labels)  # , probs_over)

    # Detection of beginning of music
    for j in range(len(test_generator.files_length)):
        music_pos, music_dur = process_label.counting(soft_labels[j], 'M')
        print('Music detected in:')
        for i in range(len(music_pos)):
            if overlap == 0:
                print('Beginning: ', (music_pos[i] * separation) // 60, 'min ',
                      int((music_pos[i] * separation) % 60),
                      'seg - Duration: ', music_dur[i] * separation)
            else:
                print('Beginning: ', int((music_pos[i] * overlap) // 60), 'min ',
                      int((music_pos[i] * overlap) % 60),
                      'seg - Duration: ', music_dur[i] * overlap)


def main(argv):
    # Utility main to load flags
    try:
        argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
