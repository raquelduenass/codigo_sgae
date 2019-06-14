import gflags
import numpy as np
import os
import sys
import csv
import utils
import utils_demo
import process_label
import pandas as pd
from keras import backend as k
from common_flags import FLAGS

# Constants
TEST_PHASE = 1
CLASSES = ['music', 'music_speech', 'speech', 'noise']
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
    weights_load_path = os.path.abspath(FLAGS.weights_filename)
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except ImportError:
        print("Impossible to find weight path. Returning untrained model")

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Get predictions and ground truth
    nb_batches = int(np.ceil(test_generator.samples / FLAGS.batch_size))
    prob_per_class = utils.compute_predictions(model, test_generator, nb_batches, verbose=1)

    # Save NN output
    with open(os.path.join(FLAGS.experiment_root_directory, "probs_out.csv"), "w") as f:
        wr = csv.writer(f)
        wr.writerows(prob_per_class)

    # Process labels
    data = pd.read_csv(os.path.join(FLAGS.demo_path, 'data.csv'))
    predicted_labels = process_label.predicted_label_process(prob_per_class, test_generator.files_length)
    real_labels = process_label.real_label_process(data['ground_truth'].tolist(), test_generator.samples,
                                                   test_generator.files_length)

    # Save predicted and softened labels as a dictionary
    labels_dict = {'predicted_labels': predicted_labels, 'real_labels': real_labels}
    utils.write_to_file(labels_dict, os.path.join(FLAGS.experiment_root_directory,
                                                  'demo_predicted_and_soft_labels.json'))
    flat_real = [item for sublist in real_labels for item in sublist]
    flat_predicted = [item for sublist in predicted_labels for item in sublist]
    utils.plot_confusion_matrix('demo', FLAGS.experiment_root_directory, flat_real,
                                flat_predicted, CLASSES, normalize=True)

    # Metrics and boundaries of music
    for j in range(len(test_generator.file_names)):
        print('File: '+str(test_generator.file_names[j]))
        process_label.show_results(real_labels[j], predicted_labels[j])
        process_label.plot_output(real_labels[j], 1, test_generator.file_names[j])
        process_label.plot_output(predicted_labels[j], 2, test_generator.file_names[j])


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
