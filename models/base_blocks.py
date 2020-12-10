from keras.layers import Conv2D, Activation, BatchNormalization, Input, MaxPool2D, Conv2DTranspose, \
    Concatenate, Dense, Flatten
from keras.models import Model
import keras.backend as K


def create_block(input, chs):
    # Convolution block of 2 layers
    x = input
    for i in range(1):
        x = Conv2D(chs, 3, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
    return x


def get_autoencoder():
    input = Input((32, 32, 3))

    # Encoder
    block1 = create_block(input, 32)
    x = MaxPool2D(2)(block1)
    # block2 = create_block(x, 64)
    # x = MaxPool2D(2)(block2)
    # block3 = create_block(x, 64)
    # x = MaxPool2D(2)(block3)
    # block4 = create_block(x, 128)

    # Middle
    # x = MaxPool2D(2)(block2)
    middle = create_block(x, 32)

    # Decoder
    # x = Conv2DTranspose(128, kernel_size=2, strides=2)(middle)
    # x = Concatenate()([block4, x])
    # x = create_block(x, 128)
    # x = Conv2DTranspose(64, kernel_size=2, strides=2)(x)
    # x = Concatenate()([block3, x])
    # x = create_block(x, 64)
    # x = Conv2DTranspose(64, kernel_size=2, strides=2)(middle)
    # x = Concatenate()([block1, x])
    # x = create_block(x, 64)
    # x = Conv2DTranspose(32, kernel_size=2, strides=2)(x)
    # x = Concatenate()([block1, x])
    # x = create_block(x, 32)

    # output
    # x = Conv2D(3, 1)(x)
    # output = Activation("sigmoid")(x)

    return Model(input, middle), None#, Model(input, output)


def get_projector(input_dims, dims=500):
    inputs = Input(input_dims)
    x = Flatten()(inputs)
    x = Dense(2048, activation='relu')(x)
    x = Dense(dims)(x)
    return Model(inputs, x)
