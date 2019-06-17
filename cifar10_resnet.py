from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, Concatenate
from keras.regularizers import l2
from keras.models import Model
from common_flags import FLAGS


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = FLAGS.initial_lr
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def res_net_layer(inputs,
                  num_filters=16,
                  kernel_size=3,
                  activation='relu',
                  batch_normalization=True,
                  convolution_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Convolution2D number of filters
        kernel_size (int): Convolution2D square kernel dimensions
        strides (int): Convolution2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        convolution_first (bool): convolution-bn-activation (True) or
            activation-bn-convolution (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    convolution = Conv2D(num_filters,
                         kernel_size=kernel_size,
                         strides=1,
                         padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-4))

    x = inputs
    if convolution_first:
        x = convolution(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = convolution(x)
    return x


def base_res_net_v1(input_shape, depth):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Convolution2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (down-sampled)
    by a convolution layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolution 2D layers
    # Returns
        model (Model): model instance
    """
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(input_shape)

    x = res_net_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            y = res_net_layer(inputs=x,
                              num_filters=num_filters,
                              activation='relu')
            y = res_net_layer(inputs=y,
                              num_filters=num_filters)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = res_net_layer(inputs=x,
                                  num_filters=num_filters,
                                  kernel_size=1,
                                  batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    model = Model(inputs=inputs, outputs=y)
    return model


def base_res_net_v2(input_shape, depth):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Convolution2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Convolution2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (down-sampled)
    by a convolution layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    convolution1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolution layers
        num_classes (int): number of classes
    # Returns
        model (Model): model instance
    """
    # Start model definition.
    num_filters_in = 16
    num_filters_out = 0
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)

    # v2 performs Convolution 2D with BN-ReLU on input before splitting into 2 paths
    x = res_net_layer(inputs=inputs,
                      num_filters=num_filters_in,
                      activation='relu',
                      convolution_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2

            # bottleneck residual unit
            y = res_net_layer(inputs=x,
                              num_filters=num_filters_in,
                              kernel_size=1,
                              activation=activation,
                              batch_normalization=batch_normalization,
                              convolution_first=False)
            y = res_net_layer(inputs=y,
                              num_filters=num_filters_in,
                              activation='relu',
                              convolution_first=False)
            y = res_net_layer(inputs=y,
                              num_filters=num_filters_out,
                              kernel_size=1,
                              activation='relu',
                              convolution_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = res_net_layer(inputs=x,
                                  num_filters=num_filters_out,
                                  kernel_size=1,
                                  batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    model = Model(inputs=inputs, outputs=y)
    return model


def res_net(input_shape, depth):
    """
    Model with shared layers based on ResNet architecture
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolution 2D layers
    # Returns
        model (Model): model instance
    """
    base_model = []

    # Output dimension
    if FLAGS.f_output == 'sigmoid':
        num_classes = FLAGS.num_classes - 1
    else:
        num_classes = FLAGS.num_classes

    # Create simple model
    if FLAGS.version == 1:
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        base_model = base_res_net_v1(input_shape, depth)
    elif FLAGS.version == 2:
        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        base_model = base_res_net_v2(input_shape, depth)

    # Simple model architecture
    if FLAGS.structure == 'simple':
        inputs = Input(shape=input_shape)
        y = base_model(inputs)

    # Parallel model architecture
    elif FLAGS.structure == 'complex':
        inputs, features = [[]] * FLAGS.wind_len, [[]] * FLAGS.wind_len
        # Reuse model
        for i in range(FLAGS.wind_len):
            inputs[i] = Input(input_shape)
            features[i] = base_model(inputs[i])

        y = Concatenate()(features)

    # Model output
    outputs = Dense(num_classes,
                    activation=FLAGS.f_output,
                    kernel_initializer='he_normal')(y)
    model = Model(inputs=inputs, outputs=outputs)
    return model
