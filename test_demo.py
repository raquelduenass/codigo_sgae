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
from sklearn import metrics

# Constants
TEST_PHASE = 1
CLASSES = ['music', 'music_speech', 'speech', 'noise']


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
    data = pd.read_csv(os.path.join(FLAGS.demo_path, 'data.csv'))
    real_labels = data['ground_truth'].tolist()

    # Real labels (ground truth)
    real_labels = np.argmax(process_label.real_label_process(real_labels, len(prob_per_class),
                                                             [len(prob_per_class)]), axis=-1).T
    # Predicted probabilities
    predicted_labels = np.argmax(process_label.predict(prob_per_class, FLAGS.threshold), axis=-1).T
    probabilities_per_class = np.asarray(prob_per_class)

    # Evaluate predictions: Average accuracy and highest errors
    print("-----------------------------------------------")
    print("Evaluation of classification:")
    print('Average accuracy = ', metrics.accuracy_score(real_labels, predicted_labels))
    print('Precision = ', metrics.precision_score(real_labels, predicted_labels, average='weighted'))
    print('Recall = ', metrics.recall_score(real_labels, predicted_labels, average='weighted'))
    print('F-score = ', metrics.f1_score(real_labels, predicted_labels, average='weighted'))
    print("-----------------------------------------------")

    # Save predicted and real labels as a dictionary
    labels_dict = {'probabilities': probabilities_per_class.tolist(),
                   'predicted_labels': predicted_labels.tolist(),
                   'real_labels': real_labels.tolist()}
    utils.write_to_file(labels_dict, os.path.join(FLAGS.experiment_root_directory,
                                                  'demo_predicted_and_real_labels.json'))

    # Visualize confusion matrix
    utils.plot_confusion_matrix('demo', FLAGS.experiment_root_directory, real_labels,
                                predicted_labels, CLASSES, normalize=True)

    # Turn to music detection
    real_labels = real_labels.tolist()
    real_labels = [real_labels[i][0] for i in range(len(real_labels))]
    real_labels = process_label.music_detection(real_labels)
    predicted_labels = process_label.music_detection(predicted_labels)

    # Evaluate predictions: Average accuracy and highest errors
    print("-----------------------------------------------")
    print("Evaluation of detection:")
    print('Average accuracy = ', metrics.accuracy_score(real_labels, predicted_labels))
    print('Precision = ', metrics.precision_score(real_labels, predicted_labels, average='weighted'))
    print('Recall = ', metrics.recall_score(real_labels, predicted_labels, average='weighted'))
    print('F-score = ', metrics.f1_score(real_labels, predicted_labels, average='weighted'))
    print("-----------------------------------------------")

    real_labels = process_label.separate_labels(real_labels, test_generator.files_length)
    predicted_labels = process_label.separate_labels(predicted_labels, test_generator.files_length)

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
