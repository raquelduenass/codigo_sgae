import gflags
import numpy as np
import os
import sys
import utils
import demo_utils
from sklearn import metrics
from keras import backend as K
from common_flags import FLAGS 
from scipy.signal import medfilt

# Constants
TEST_PHASE = 1
CLASSES = ['MH','M','H']
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
os.environ["PATH"] += os.pathsep + 'C:/Users/rds/Downloads/ffmpeg/bin'

def _main():

    # Set testing mode (dropout/batchnormalization)
    K.set_learning_phase(TEST_PHASE)
    
    # Output dimension (2 classes)
    num_classes = 3
    sr = 22050
    separation = 2
    power=2

    # Generate testing data
    test_datagen = demo_utils.DataGenerator(rescale=1./255)
    
    # Iterator object containing testing data to be generated batch by batch
    test_generator = test_datagen.flow_from_directory(num_classes,
                                                      power,
                                                      sr,
                                                      separation,
                                                      shuffle = False,
                                                      target_size=(FLAGS.img_height, FLAGS.img_width),
                                                      batch_size = FLAGS.batch_size)

    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    model = utils.jsonToModel(json_model_path)

    # Load weights
    weights_load_path = os.path.abspath('./models/test_3/weights_019.h5')
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
    probs_per_class, ground_truth = utils.compute_predictions_and_gt(
            model, test_generator, nb_batches, verbose = 1)
    
    # Prediced labels
    silence_labels = test_generator.silence_labels
    classes = [CLASSES[i] for i in np.argmax(probs_per_class, axis=-1)]
    pred_labels = utils.join_labels(classes,silence_labels)
    real_labels = [CLASSES[i] for i in np.argmax(ground_truth, axis=-1)]
    
    # Standardize real labels
    for i in range(len(real_labels)):
        if not (real_labels[i]== 'M'):
            real_labels[i]='NM'
            
    # Save predicted and real steerings as a dictionary
    labels_dict = {'pred_labels': pred_labels,
                  'real_labels': real_labels}
    utils.write_to_file(labels_dict, os.path.join(FLAGS.experiment_rootdir,
                                               'demo_predicted_and_real_labels.json'))
            
    # Accuracy before softening
    ave_accuracy = metrics.accuracy_score(real_labels,pred_labels)
    print('Initial accuracy: ', ave_accuracy)
    
    # Class softening
    dct = {'MH':0,'M':1,'H':2,'S':3}
    new_dct = {0:'M',1:'M',2:'NM', 3:'NM'}
    labels = list(map(dct.get, pred_labels))
    softened = medfilt(labels, 9)
    softened = medfilt(softened, 15)
    soft_labels = list(map(new_dct.get, softened))
    
    # Accuracy after softening
    ave_accuracy = metrics.accuracy_score(real_labels,soft_labels)
    print('Softening accuracy: ', ave_accuracy)
    
    # Detection of beginning of music
    music_pos, music_dur = utils.counting(soft_labels, 'M')
    print('Music detected in:')
    for i in range(len(music_pos)):
        print('Inicio: ',(music_pos[i]*separation)//60, 'min ', int((music_pos[i]*separation)%60),
              'seg - Duración: ', music_dur[i]*separation)
    
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