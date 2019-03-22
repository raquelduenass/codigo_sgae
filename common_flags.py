import gflags

FLAGS = gflags.FLAGS

# Random seed
gflags.DEFINE_bool('random_seed', True, 'Random seed')

# Input
gflags.DEFINE_integer('img_width', 100, 'Target Image Width')
gflags.DEFINE_integer('img_height', 100, 'Target Image Height')
gflags.DEFINE_string('img_mode', "grayscale", 'Load mode for images, either '
                     'rgb or grayscale')

# Training parameters
gflags.DEFINE_integer('batch_size', 128, 'Batch size in training and evaluation')
gflags.DEFINE_integer('epochs', 20, 'Number of epochs for training')
gflags.DEFINE_integer('initial_epoch', 0, 'Initial epoch to start training')
gflags.DEFINE_float('initial_lr', 1e-4, 'Initial learning rate for adam')

# Files
gflags.DEFINE_string('experiment_rootdir', "./models/test_4", 'Folder '
                     ' containing all the logs, model weights and results')
gflags.DEFINE_string('data_path', "./../data_sgae/spectrograms",
                     'Folder containing the whole dataset')
gflags.DEFINE_string('demo_path', "../data_sgae/muspeak",
                     'Folder containing the demo dataset')
gflags.DEFINE_string('exp_name', "exp_1", 'Name of the experiment'
                     ' to be processed')

# Model
gflags.DEFINE_bool('restore_model', True, 'Whether to restore a trained'
                   ' model for training')
gflags.DEFINE_string('weights_fname', './models/test_3/weights_019.h5',
                     '(Relative) filename of model weights')
gflags.DEFINE_string('initial_weights','./models/test_3/weights_019.h5',
                     '(Relative) filename of model initial training weights')
gflags.DEFINE_string('json_model_fname', "model_struct.json",
                          'Model struct json serialization, filename')