import gflags

FLAGS = gflags.FLAGS

# Random seed
gflags.DEFINE_bool('random_seed', True, 'Random seed')

# Input
gflags.DEFINE_integer('img_width', 49, 'Target Image Width')
gflags.DEFINE_integer('img_height', 64, 'Target Image Height')
gflags.DEFINE_integer('num_classes', 4, 'Number of classes in data set')
gflags.DEFINE_float('separation', 0.96, 'Duration in seconds of the represented spectrogram')
gflags.DEFINE_integer('sr', 22050, 'Sample rate imposed to audio')
gflags.DEFINE_integer('power', 2, 'Type of value represented in spectrogram')
gflags.DEFINE_float('overlap', 0.5, 'Time overlap among samples in demo')
gflags.DEFINE_integer('wind_len', 5, 'Window length of temporal filtering')

# Training parameters
gflags.DEFINE_integer('n', 1, 'depth of network')
gflags.DEFINE_integer('version', 2, 'ResNet block type')
gflags.DEFINE_string('structure', 'complex', 'Parallel layers ("complex") or not ("simple")')
gflags.DEFINE_boolean('from_audio', False, 'Input to the net from audio or image')
gflags.DEFINE_integer('batch_size', 32, 'Batch size in training and evaluation')
gflags.DEFINE_integer('epochs', 10, 'Number of epochs for training')
gflags.DEFINE_integer('initial_epoch', 0, 'Initial epoch to start training')
gflags.DEFINE_float('initial_lr', 1e-4, 'Initial learning rate for adam')
gflags.DEFINE_string('f_output', 'sigmoid', 'Output function')
gflags.DEFINE_float('threshold', 0.4, 'Minimum value for the sigmoid output to be active')

# Files
gflags.DEFINE_string('experiment_root_directory', "./models/test_12", 'Folder '
                     ' containing all the logs, model weights and results')
gflags.DEFINE_string('data_path', "../../databases/df_spec",
                     'Folder containing the whole data set')
gflags.DEFINE_string('demo_path', "../../databases/muspeak",
                     'Folder containing the demo data set')
gflags.DEFINE_string('exp_name', "exp_1", 'Name of the experiment'
                     ' to be processed')

# Model
gflags.DEFINE_bool('restore_model', False, 'Whether to restore a trained'
                   ' model for training')
gflags.DEFINE_string('weights_filename', './models/test_12/weights_010.h5',
                     '(Relative) filename of model weights')
gflags.DEFINE_string('initial_weights', './models/test_14/weights_004.h5',
                     '(Relative) filename of model initial training weights')
gflags.DEFINE_string('json_model_filename', "model_structure.json",
                     'Model structure json serialization, filename')
