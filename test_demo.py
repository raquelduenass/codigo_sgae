import gflags
import numpy as np
import os
import sys
import utils
import demo_utils
from keras import backend as k
from common_flags import FLAGS

# Constants
TEST_PHASE = 1
CLASSES = ['M', 'MH', 'N', 'H']
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
    overlap = 0
    wind_len = 5

    # Generate testing data
    test_datagen = demo_utils.DataGenerator(rescale=1./255)
    
    # Iterator object containing testing data to be generated batch by batch
    test_generator = test_datagen.flow_from_directory(num_classes,
                                                      power,
                                                      sr,
                                                      separation,
                                                      overlap,
                                                      shuffle=False,
                                                      target_size=(FLAGS.img_height, FLAGS.img_width),
                                                      batch_size=FLAGS.batch_size)

    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    model = utils.json_to_model(json_model_path)

    # Load weights
    weights_load_path = os.path.abspath('./models/test_4/weights_011.h5')
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
    probs_per_class = utils.compute_predictions(
            model, test_generator, nb_batches, verbose=1)
    
    # Predicted labels
    silence_labels = test_generator.silence_labels
    classes = [CLASSES[i] for i in np.argmax(probs_per_class, axis=-1)]
    pred_labels = utils.join_labels(classes, silence_labels)
    
    # Class softening
    soft_labels = utils.soft_max(pred_labels, wind_len)
    
    # Detection of beginning of music
    music_pos, music_dur = utils.counting(soft_labels, 'M')
    print('Music detected in:')
    for i in range(len(music_pos)):
        if overlap == 0:
            print('Beginning: ', (music_pos[i] * separation) // 60, 'min ',
                  int((music_pos[i] * separation) % 60),
                  'seg - Duration: ', music_dur[i] * separation)
        else:
            print('Beginning: ', (music_pos[i]*overlap*separation)//60, 'min ',
                  int((music_pos[i]*overlap*separation) % 60),
                  'seg - Duration: ', music_dur[i]*separation)


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
