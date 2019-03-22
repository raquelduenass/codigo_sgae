import tensorflow as tf
import numpy as np
import os
import sys
import gflags

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras import backend as k

import logz
import nets
import cifar10_resnet
import utils
import data_utils
import log_utils
from common_flags import FLAGS
from time import time, strftime, localtime
import database_process

# Constants
TRAIN_PHASE = 1

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
os.environ["PATH"] += os.pathsep + 'C:/Users/rds/Downloads/ffmpeg/bin'


def get_model(img_height, img_width, output_dim, weights_path):
    """
    Initialize model.
    # Arguments
       img_width: Target image width.
       img_height: Target image height.
       num_img: Target images per block
       output_dim: Dimension of model output (number of classes).
       weights_path: Path to pre-trained model.
    # Returns
       model: A Model instance.
    """
    model = nets.resnet50(img_height, img_width, 1, output_dim)
    if weights_path:
        try:
            model.load_weights(weights_path)
            print("Loaded model from {}".format(weights_path))
        except ImportError:
            print("Impossible to find weight path. Returning untrained model")
    return model


def get_model_res_net(n, version, img_height, img_width, output_dim, weights_path):
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
    # Returns
       model: A Model instance.
    """
    
    input_shape = (img_height, img_width, 1)
    
    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
        model = cifar10_resnet.resnet_v1(input_shape=input_shape, depth=depth, num_classes=output_dim)
    else:
        depth = n * 9 + 2
        model = cifar10_resnet.resnet_v2(input_shape=input_shape, depth=depth, num_classes=output_dim)        

    # Model name, depth and version
    model_type = 'ResNet%dv%d' % (depth, version)
    print(model_type)
    print(model.summary())

    if weights_path:
        try:
            model.load_weights(weights_path)
            print("Loaded model from {}".format(weights_path))
        except ImportError:
            print("Impossible to find weight path. Returning untrained model")

    return model


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
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=cifar10_resnet.lr_schedule(0)),
                  metrics=['categorical_accuracy'])

    # Save model with the lowest validation loss
    weights_path = os.path.join(FLAGS.experiment_rootdir, 'weights_{epoch:03d}.h5')
    write_best_model = ModelCheckpoint(filepath=weights_path, monitor='val_loss',
                                       save_best_only=True, save_weights_only=True)

    # Save training and validation losses.
    logz.configure_output_dir(FLAGS.experiment_rootdir)
    save_model_and_loss = log_utils.MyCallback(filepath=FLAGS.experiment_rootdir)

    # Train model
    steps_per_epoch = int(np.ceil(train_data_generator.samples / FLAGS.batch_size))
    validation_steps = int(np.ceil(val_data_generator.samples / FLAGS.batch_size))-1
    
    lr_scheduler = LearningRateScheduler(cifar10_resnet.lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    
    # Save Tensor board information
    str_time = strftime("%Y%b%d_%Hh%Mm%Ss", localtime(time()))
    tensor_board = TensorBoard(log_dir="logs/{}".format(str_time), histogram_freq=0)
    callbacks = [write_best_model, save_model_and_loss, lr_reducer, lr_scheduler, tensor_board]

    model.fit_generator(train_data_generator,
                        epochs=FLAGS.epochs, steps_per_epoch=steps_per_epoch,
                        callbacks=callbacks,
                        validation_data=val_data_generator,
                        validation_steps=validation_steps,
                        initial_epoch=initial_epoch)


def _main():
    
    # Set random seed
    if FLAGS.random_seed:
        seed = np.random.randint(0, 2*31-1)
    else:
        seed = 5
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    # Parameters
    num_classes = 4
    flag_data = 0
    separation = 2
    # overlap = 0
    music_pct = 0.8
    n = 1
    version = 2  # 1 o 2
    
    if flag_data:
        database_process.classes_combination(FLAGS.data_path, False, [0, 1], music_pct)
        database_process.create_database(FLAGS.data_path, True, separation)
        database_process.create_database(FLAGS.demo_path, False, separation)
        database_process.labels_demo(FLAGS.demo_path, 'labels.txt', num_classes)

    # Set training phase
    k.set_learning_phase(TRAIN_PHASE)

    # Create the experiment root dir if not already there:
    # create a model if the name of the one in the parameters doesn't exist
    if not os.path.exists(FLAGS.experiment_rootdir):
        os.makedirs(FLAGS.experiment_rootdir)
        
    # Split the data into training, validation and test sets
    if FLAGS.initial_epoch == 0:
        data_utils.cross_val_create(FLAGS.data_path)
    
    # Input image dimensions
    img_width, img_height = FLAGS.img_width, FLAGS.img_height

    # Generate training data with real-time augmentation
    train_data_gen = data_utils.DataGenerator(rescale=1./255)
    
    # Iterator object containing training data to be generated batch by batch
    train_generator = train_data_gen.flow_from_directory('train',
                                                         shuffle=True,
                                                         target_size=(img_height, img_width),
                                                         batch_size=FLAGS.batch_size)
    
    # Check if the number of classes in data corresponds to the one specified
    assert train_generator.num_classes == num_classes, \
        " Not matching output dimensions in training data."

    # Generate validation data with real-time augmentation
    val_data_gen = data_utils.DataGenerator(rescale=1./255)
    
    # Iterator object containing validation data to be generated batch by batch
    val_generator = val_data_gen.flow_from_directory('val',
                                                     shuffle=False,
                                                     target_size=(img_height, img_width),
                                                     batch_size=FLAGS.batch_size)

    # Check if the number of classes in data corresponds to the one specified
    assert val_generator.num_classes == num_classes, \
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
    model = get_model_res_net(n, version, img_height, img_width, num_classes, weights_path)

    # Serialize model into json
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    utils.modelToJson(model, json_model_path)

    # Train model
    train_model(train_generator, val_generator, model, initial_epoch)
    
    # Plot training and validation losses
    utils.plot_loss(FLAGS.experiment_rootdir)


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
