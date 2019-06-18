import gflags
import os
import sys
import utils
import utils_data
import numpy as np
import process_label
from sklearn import metrics
from keras import backend as k
from common_flags import FLAGS 

# Constants
TEST_PHASE = 1
CLASSES = ['music', 'music_speech', 'speech', 'noise']


def compute_highest_classification_errors(predicted_probabilities, real_labels, n_errors=20):
    """
    Compute the 'n_errors' highest errors predicted by the network

    # Arguments
       pred_probs: predicted probabilities by the network.
       real_labels: real labels (ground truth).
       n_errors: Number of samples with highest error to be returned.
       
    # Returns
       highest_errors: Indexes of the samples with highest errors.
    """
    assert np.all(predicted_probabilities.shape == real_labels.shape)
    dist = abs(predicted_probabilities - 1)
    highest_errors = dist.argsort()[-n_errors:][::-1]
    return highest_errors


def evaluate_classification(predicted_probabilities, predicted_labels, real_labels):
    """
    Evaluate some classification metrics. Compute average accuracy and highest
    errors.
    # Arguments
       pred_probs: predicted probabilities by the network.
       pred_labels: predicted labels by the network.
       real_labels: real labels (ground truth).
    # Returns
       dictionary: dictionary containing the evaluated classification metrics
    """
    # Compute average accuracy
    ave_accuracy = metrics.accuracy_score(real_labels, predicted_labels)
    print('Average accuracy = ', ave_accuracy)
    
    # Compute highest errors
    highest_errors = compute_highest_classification_errors(predicted_probabilities, real_labels,
                                                           n_errors=20)
    
    # Return accuracy and highest errors in a dictionary
    dictionary = {"ave_accuracy": ave_accuracy,
                  "highest_errors": highest_errors.tolist()}
    return dictionary


def _main():

    # Set testing mode (dropout/batch normalization)
    k.set_learning_phase(TEST_PHASE)

    # Generate testing data
    test_data_gen = utils_data.DataGenerator(rescale=1./255)
    
    # Iterator object containing testing data to be generated batch by batch
    test_generator = test_data_gen.flow_from_directory('test', shuffle=False,
                                                       target_size=(FLAGS.img_height, FLAGS.img_width),
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
    model.compile(loss='mse', optimizer='adam')

    # Get predictions and ground truth
    n_samples = test_generator.samples
    nb_batches = int(np.ceil(n_samples / FLAGS.batch_size))
    probabilities_per_class, ground_truth = utils.compute_predictions_and_gt(
            model, test_generator, nb_batches, verbose=1)

    saving_dict = {'probabilities': probabilities_per_class.tolist(),
                   'ground_truth': ground_truth.tolist()}
    utils.write_to_file(saving_dict, os.path.join(FLAGS.experiment_root_directory,
                                                  'save_predicted_and_real_labels.json'))

    # Real labels (ground truth)
    real_labels = np.argmax(process_label.sigmoid_to_softmax(ground_truth.tolist()), axis=-1)

    # Predicted probabilities
    predicted_labels = np.argmax(process_label.predict(probabilities_per_class, FLAGS.threshold), axis=-1)

    # Evaluate predictions: Average accuracy and highest errors
    print("-----------------------------------------------")
    print("Evaluation:")
    ave_accuracy = metrics.accuracy_score(real_labels, predicted_labels)
    print('Average accuracy = ', ave_accuracy)
    print("-----------------------------------------------")

    # Save evaluation
    utils.write_to_file([ave_accuracy], os.path.join(FLAGS.experiment_root_directory, 'test_results.json'))

    # Save predicted and real labels as a dictionary
    labels_dict = {'probabilities': probabilities_per_class.tolist(),
                   'predicted_labels': predicted_labels.tolist(),
                   'real_labels': real_labels.tolist()}
    utils.write_to_file(labels_dict, os.path.join(FLAGS.experiment_root_directory,
                                                  'predicted_and_real_labels.json'))
                                               
    # Visualize confusion matrix                                           
    utils.plot_confusion_matrix('test', FLAGS.experiment_root_directory, real_labels,
                                predicted_labels, CLASSES, normalize=True)


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
