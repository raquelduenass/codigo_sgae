import tensorflow as tf
import numpy as np
import os
import sys
import gflags
import logz
# import nets
import cifar10_resnet
import utils
import utils_data
import utils_data_audio
import log_utils
from common_flags import FLAGS
from time import time, strftime, localtime
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras import backend as k


# Constants
TRAIN_PHASE = 1

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
os.environ["PATH"] += os.pathsep + 'C:/Users/rds/Downloads/ffmpeg/bin'


def get_model_res_net(img_height, img_width, weights_path):
    """
    Initialize model.
    # Arguments
       n: parameter that determines the net depth.
       version: 1 for ResNet v1 or 2 for v2.
       img_width: Target image width.
       img_height: Target image height.
       num_img: Target images per block
       output_dim: Dimension of model output (number of classes).
       weights_path: Path to pre-trained model.
       f_output: function of the last network layer ('sigmoid' or 'softmax')
       structure: network architecture ('simple' with one spectrogram input or 'complex'
                  with many inputs and filtering implied)
    # Returns
       model: A Model instance.
    """
    
    input_shape = (img_height, img_width, 1)
    
    # Computed depth from supplied model parameter n
    if FLAGS.version == 1:
        depth = FLAGS.n * 6 + 2
    else:
        depth = FLAGS.n * 9 + 2

    if FLAGS.structure == 'simple':
        model = cifar10_resnet.res_net(input_shape=input_shape, depth=depth)
    else:
        model = cifar10_resnet.comb_res_net(input_shape=input_shape, depth=depth)

    # Model name, depth and version
    model_type = 'ResNet%dv%d' % (depth, FLAGS.version)
    print(model_type)
    print(model.summary())

    if weights_path:
        try:
            model.load_weights(weights_path)
            print("Loaded model from {}".format(weights_path))
        except ImportError:
            print("Impossible to find weight path. Returning untrained model")

    return model


def many_generator(generator):
    """
    Provides the multiple network inputs from the batch data
    """
    # TODO: Estandarizar al tamaño de FLAGS.wind_len
    while True:
        first, second, third, forth, fifth = [], [], [], [], []
        x = generator.next()
        for i in range(FLAGS.batch_size):
            first.append(x[0][i][0])
            second.append(x[0][i][1])
            third.append(x[0][i][2])
            forth.append(x[0][i][3])
            fifth.append(x[0][i][4])
        # Yield both images and their mutual label
        yield [np.asarray(first), np.asarray(second), np.asarray(third),
               np.asarray(forth), np.asarray(fifth)], x[1]


def train_model(train_data_generator, val_data_generator, model, initial_epoch):
    """
    Model training.
    # Arguments
       train_data_generator: Training data generated batch by batch.
       val_data_generator: Validation data generated batch by batch.
       model: A Model instance.
       initial_epoch: Epoch from which training starts.
    """
    # Configure training process
    model.compile(loss='mse',
                  optimizer=Adam(lr=cifar10_resnet.lr_schedule(0)),
                  metrics=['mse'])

    # Save model with the lowest validation loss
    weights_path = os.path.join(FLAGS.experiment_root_directory, 'weights_{epoch:03d}.h5')
    write_best_model = ModelCheckpoint(filepath=weights_path, monitor='val_loss',
                                       save_best_only=True, save_weights_only=True)

    # Save training and validation losses.
    logz.configure_output_dir(FLAGS.experiment_root_directory)
    save_model_and_loss = log_utils.MyCallback(filepath=FLAGS.experiment_root_directory)

    # Train model
    steps_per_epoch = int(np.ceil(train_data_generator.samples / FLAGS.batch_size))
    validation_steps = int(np.ceil(val_data_generator.samples / FLAGS.batch_size))-1

    # Learning rate
    lr_scheduler = LearningRateScheduler(cifar10_resnet.lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0,
                                   patience=5, min_lr=0.5e-6)
    
    # Save Tensor board information
    str_time = strftime("%Y%b%d_%Hh%Mm%Ss", localtime(time()))
    tensor_board = TensorBoard(log_dir="logs/{}".format(str_time), histogram_freq=0)
    callbacks = [write_best_model, save_model_and_loss, lr_reducer, lr_scheduler, tensor_board]

    if FLAGS.structure == 'complex':
        model.fit_generator(many_generator(train_data_generator),
                            epochs=FLAGS.epochs, steps_per_epoch=steps_per_epoch,
                            callbacks=callbacks,
                            validation_data=many_generator(val_data_generator),
                            validation_steps=validation_steps,
                            initial_epoch=initial_epoch,
                            max_queue_size=30,
                            workers=0,
                            use_multiprocessing=False)
    else:
        model.fit_generator(train_data_generator,
                            epochs=FLAGS.epochs, steps_per_epoch=steps_per_epoch,
                            callbacks=callbacks,
                            validation_data=val_data_generator,
                            validation_steps=validation_steps,
                            initial_epoch=initial_epoch,
                            max_queue_size=10,
                            workers=2,
                            use_multiprocessing=False)


def _main():
    
    # Set random seed
    if FLAGS.random_seed:
        seed = np.random.randint(0, 2*31-1)
    else:
        seed = 5
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Set training phase
    k.set_learning_phase(TRAIN_PHASE)

    # Create the experiment root dir if not already there:
    # create a model if the name of the one in the parameters doesn't exist
    if not os.path.exists(FLAGS.experiment_root_directory):
        os.makedirs(FLAGS.experiment_root_directory)
        
    # Split the data into training, validation and test sets
    if FLAGS.initial_epoch == 0:
        if FLAGS.from_audio:
            utils_data_audio.cross_val_create(FLAGS.data_path)
        else:
            utils_data.cross_val_create(FLAGS.data_path)
    
    # Input image dimensions
    img_width, img_height = FLAGS.img_width, FLAGS.img_height

    if FLAGS.from_audio:
        # Generate training data with real-time augmentation
        train_data_gen = utils_data_audio.DataGenerator(rescale=1. / 255)

        # Iterator object containing training data to be generated batch by batch
        train_generator = train_data_gen.flow_from_directory('train',
                                                             shuffle=True,
                                                             target_size=(img_height, img_width),
                                                             batch_size=FLAGS.batch_size)

        # Generate validation data with real-time augmentation
        val_data_gen = utils_data_audio.DataGenerator(rescale=1. / 255)

        # Iterator object containing validation data to be generated batch by batch
        val_generator = val_data_gen.flow_from_directory('val',
                                                         shuffle=False,
                                                         target_size=(img_height, img_width),
                                                         batch_size=FLAGS.batch_size)
    else:
        # Generate training data with real-time augmentation
        train_data_gen = utils_data.DataGenerator(rescale=1. / 255)

        # Iterator object containing training data to be generated batch by batch
        train_generator = train_data_gen.flow_from_directory('train',
                                                             shuffle=True,
                                                             target_size=(img_height, img_width),
                                                             batch_size=FLAGS.batch_size)

        # Generate validation data with real-time augmentation
        val_data_gen = utils_data.DataGenerator(rescale=1. / 255)

        # Iterator object containing validation data to be generated batch by batch
        val_generator = val_data_gen.flow_from_directory('val',
                                                         shuffle=False,
                                                         target_size=(img_height, img_width),
                                                         batch_size=FLAGS.batch_size)

    # Check if the number of classes in data corresponds to the one specified
    assert train_generator.num_classes == FLAGS.num_classes, \
        " Not matching output dimensions in training data."

    # Check if the number of classes in data corresponds to the one specified
    assert val_generator.num_classes == FLAGS.num_classes, \
        " Not matching output dimensions in validation data."
          
    # Weights to restore
    weights_path = FLAGS.initial_weights
    
    # Epoch from which training starts
    initial_epoch = 0
    if not FLAGS.restore_model:
        # In this case weights are initialized randomly
        weights_path = None
    else:
        # In this case weights are initialized as specified in pre-trained model
        initial_epoch = FLAGS.initial_epoch

    # Define model
    model = get_model_res_net(img_height, img_width, weights_path)

    # Serialize model into json
    json_model_path = os.path.join(FLAGS.experiment_root_directory, FLAGS.json_model_filename)
    utils.model_to_json(model, json_model_path)

    # Train model
    train_model(train_generator, val_generator, model, initial_epoch)
    
    # Plot training and validation losses
    utils.plot_loss(FLAGS.experiment_root_directory)


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
