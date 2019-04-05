import gflags

FLAGS = gflags.FLAGS

# Random seed
gflags.DEFINE_bool('random_seed', True, 'Random seed')

# Input
gflags.DEFINE_integer('img_width', 173, 'Target Image Width')
gflags.DEFINE_integer('img_height', 96, 'Target Image Height')
gflags.DEFINE_string('img_mode', "grayscale", 'Load mode for images, either '
                     'rgb or gray scale')

# Training parameters
gflags.DEFINE_integer('batch_size', 32, 'Batch size in training and evaluation')
gflags.DEFINE_integer('epochs', 10, 'Number of epochs for training')
gflags.DEFINE_integer('initial_epoch', 0, 'Initial epoch to start training')
gflags.DEFINE_float('initial_lr', 1e-4, 'Initial learning rate for adam')

# Files
gflags.DEFINE_string('experiment_rootdir', "./models/test_5", 'Folder '
                     ' containing all the logs, model weights and results')
gflags.DEFINE_string('data_path', "./../data_sgae/spectrograms_npy",
                     'Folder containing the whole data set')
gflags.DEFINE_string('demo_path', "../data_sgae/muspeak",
                     'Folder containing the demo dataset')
gflags.DEFINE_string('exp_name', "exp_1", 'Name of the experiment'
                     ' to be processed')

# Model
gflags.DEFINE_bool('restore_model', False, 'Whether to restore a trained'
                   ' model for training')
gflags.DEFINE_string('weights_fname', './models/test_5/weights_011.h5',
                     '(Relative) filename of model weights')
gflags.DEFINE_string('initial_weights', './models/test_5/weights_011.h5',
                     '(Relative) filename of model initial training weights')
gflags.DEFINE_string('json_model_fname', "model_struct.json",
                     'Model struct json serialization, filename')
