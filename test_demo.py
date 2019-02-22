import gflags
import numpy as np
import os
import sys
import utils
import demo_utils
from sklearn import metrics
from keras import backend as K
from common_flags import FLAGS 

# Constants
TEST_PHASE = 1
CLASSES = ['M','NM']
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

def _main():

    # Set testing mode (dropout/batchnormalization)
    K.set_learning_phase(TEST_PHASE)
    
    # Output dimension (2 classes)
    num_classes = 2

    # Generate testing data
    test_datagen = demo_utils.DataGenerator(rescale=1./255)
    
    # Iterator object containing testing data to be generated batch by batch
    test_generator = test_datagen.flow_from_directory(num_classes,
                                                      shuffle = False,
                                                      target_size=(FLAGS.img_height, FLAGS.img_width),
                                                      batch_size = FLAGS.batch_size)

    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    model = utils.jsonToModel(json_model_path)

    # Load weights
    weights_load_path = os.path.abspath('./models/test_1/weights_010.h5')
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except:
        print("Impossible to find weight path. Returning untrained model")

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print('Model compiled')
    
    # Get predictions and ground truth
    n_samples = test_generator.samples
    print('Samples')
    nb_batches = int(np.ceil(n_samples / FLAGS.batch_size))
    print('N batches')
    probs_per_class, ground_truth = utils.compute_predictions_and_gt(
            model, test_generator, nb_batches, verbose = 1)
    print('Probs')
    
    # Prediced labels
    silence_labels = next(test_generator, True)
    classes = CLASSES[list(np.argmax(probs_per_class, axis=-1))]
    labels = utils.join_labels(classes,silence_labels, CLASSES)
    real_labels = np.argmax(ground_truth, axis=-1)
    
    # Class softening
    soft_labels = utils.softening(labels)
    
    # Accuracy after softening
    ave_accuracy = metrics.accuracy_score(real_labels,soft_labels)
    print('Softening accuracy: ', ave_accuracy)
    
    # Detection of beginning of music
    music_pos, music_dur = utils.counting(soft_labels, 'M')
    print('Music detected in:')
    for i in range(len(music_pos)):
        print('Inicio: ',music_pos[i]//60, 'min ', int(music_pos[i]%60), 'seg - Duracion: ', music_dur[i])
    
def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)
    _main()

if __name__ == "__main__":
    main(sys.argv)