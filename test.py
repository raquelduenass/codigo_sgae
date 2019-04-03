import gflags
import numpy as np
import os
import sys
from sklearn import metrics
from keras import backend as k
import utils
import data_utils
from common_flags import FLAGS 

# ♦ Constants
TEST_PHASE = 1
power = 2
sr = 22050
separation = 2
CLASSES = ['M', 'MH', 'H', 'R']
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
os.environ["PATH"] += os.pathsep + 'C:/Users/rds/Downloads/ffmpeg/bin'


def compute_highest_classification_errors(pred_probs, real_labels, n_errors=20):
    """
    Compute the 'n_errors' highest errors predicted by the network

    # Arguments
       pred_probs: predicted probabilities by the network.
       real_labels: real labels (ground truth).
       n_errors: Number of samples with highest error to be returned.
       
    # Returns
       highest_errors: Indexes of the samples with highest errors.
    """
    assert np.all(pred_probs.shape == real_labels.shape)
    dist = abs(pred_probs - 1)
    highest_errors = dist.argsort()[-n_errors:][::-1]
    return highest_errors


def evaluate_classification(pred_probs, pred_labels, real_labels):
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
    ave_accuracy = metrics.accuracy_score(real_labels, pred_labels)
    print('Average accuracy = ', ave_accuracy)
    
    # Compute highest errors
    highest_errors = compute_highest_classification_errors(pred_probs, real_labels, n_errors=20)
    
    # Return accuracy and highest errors in a dictionary
    dictionary = {"ave_accuracy": ave_accuracy,
                  "highest_errors": highest_errors.tolist()}
    return dictionary


def _main():

    # Set testing mode (dropout/batch normalization)
    k.set_learning_phase(TEST_PHASE)

    # Generate testing data
    test_data_gen = data_utils.DataGenerator(rescale=1./255)
    
    # Iterator object containing testing data to be generated batch by batch
    test_generator = test_data_gen.flow_from_directory('test',
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
    probs_per_class, ground_truth = utils.compute_predictions_and_gt(
            model, test_generator, nb_batches, verbose=1)
    
    # Predicted probabilities
    pred_probs = np.max(probs_per_class, axis=-1)
    # Predicted labels
    pred_labels = np.argmax(probs_per_class, axis=-1)
    # Real labels (ground truth)
    real_labels = np.argmax(ground_truth, axis=-1)
    
    # Evaluate predictions: Average accuracy and highest errors
    print("-----------------------------------------------")
    print("Evaluation:")
    evaluation = evaluate_classification(pred_probs, pred_labels, real_labels)
    print("-----------------------------------------------")
    
    # Save evaluation
    utils.write_to_file(evaluation, os.path.join(FLAGS.experiment_rootdir, 'test_results.json'))

    # Save predicted and real steerings as a dictionary
    labels_dict = {'pred_labels': pred_labels.tolist(),
                   'real_labels': real_labels.tolist()}
    utils.write_to_file(labels_dict, os.path.join(FLAGS.experiment_rootdir,
                                                  'predicted_and_real_labels.json'))
                                               
    # Visualize confusion matrix                                           
    utils.plot_confusion_matrix('test', FLAGS.experiment_rootdir, real_labels,
                                pred_labels, CLASSES, normalize=True)


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
