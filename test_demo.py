import gflags
import numpy as np
import os
import sys
from keras import backend as K
import utils
import demo_utils
from common_flags import FLAGS 

# Constants
TEST_PHASE = 1
CLASSES = ['music','no-music']
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
                                                      target_size=(FLAGS.img_height, FLAGS.img_width),
                                                      batch_size = FLAGS.batch_size)

    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    model = utils.jsonToModel(json_model_path)

    # Load weights
    weights_load_path = os.path.abspath('./models/test_5/weights_002.h5')
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except:
        print("Impossible to find weight path. Returning untrained model")

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Get predictions and ground truth
    n_samples = test_generator.samples
    nb_batches = int(np.ceil(n_samples / FLAGS.batch_size))
    probs_per_class = utils.compute_predictions(
            model, test_generator, nb_batches, verbose = 1)
    
    # Prediced labels
    classes = np.argmax(probs_per_class, axis=-1)

                                               
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