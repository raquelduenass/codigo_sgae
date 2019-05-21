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
    
    # Generate testing data
    test_data_gen = utils_demo.DataGenerator(rescale=1./255)
    
    # Iterator object containing testing data to be generated batch by batch
    test_generator = test_data_gen.flow_from_directory(shuffle=False,
                                                       batch_size=FLAGS.batch_size)

    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_root_directory, FLAGS.json_model_filename)
    model = utils.json_to_model(json_model_path)

    # Load weights
    weights_load_path = os.path.abspath('./models/test_6/weights_010.h5')
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
    classes = [CLASSES[i] for i in np.argmax(prob_per_class, axis=-1)]
    predicted_labels = process_label.join_labels(classes, test_generator.silence_labels,
                                                 test_generator.files_length)
    probabilities = process_label.separate_labels(prob_per_class, test_generator.files_length)

    # Real labels
    real = utils.file_to_list(os.path.join(FLAGS.demo_path, 'labels.txt'))
    real_labels = [[]] * len(test_generator.files_length)
    for j in range(len(test_generator.files_length)):
        real[j] = ((real[j].split('[')[1]).split(']')[0]).split(', ')
        real_labels[j] = [CLASSES[int(i)] for i in real[j]]

    # Class softening
    soft_labels = process_label.soft_max(predicted_labels, len(test_generator.files_length))

    # Save predicted and softened labels as a dictionary
    labels_dict = {'predicted_labels': predicted_labels,
                   'soft_labels': soft_labels}
    utils.write_to_file(labels_dict, os.path.join(FLAGS.experiment_root_directory,
                                                  'demo_predicted_and_soft_labels.json'))

    # Metrics and boundaries of music
    for j in range(len(test_generator.files_length)):
        print('File: '+str(test_generator.file_names[j]))
        process_label.show_metrics(real_labels[j], predicted_labels[j], soft_labels[j])
        process_label.show_detections(soft_labels[j])
        process_label.visualize_output(soft_labels[j], CLASSES, probabilities[j])


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
