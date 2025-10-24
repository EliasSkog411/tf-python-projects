import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable

import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
import shutil
import pickle

notes = input("Enter any notes for this run: ") 
model_size = 512


def conv_block(x, filters, dropout_rate=0.1):
    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x, training=True)  # MC Dropout
    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x, training=True)  # MC Dropout
    return x

def bayesian_unet(input_shape=(model_size, model_size, 3), dropout_rate=0.1):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, 16, dropout_rate)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 32, dropout_rate)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 64, dropout_rate)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 128, dropout_rate)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = conv_block(p4, 256, dropout_rate)

    # Decoder
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = conv_block(u6, 128, dropout_rate)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = conv_block(u7, 64, dropout_rate)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = conv_block(u8, 32, dropout_rate)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = conv_block(u9, 16, dropout_rate)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

#
#   Models
#
def lunet_model_with_both(input_size=(model_size, model_size, 3)): 
    inputs = layers.Input(input_size)

    # Downsampling
    c1 = layers.Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)
    c4 = layers.Dropout(0.2)(c4)
    c4 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)

    # Upsampling
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation('relu')(c6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation('relu')(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Activation('relu')(c7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Activation('relu')(c7)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Activation('relu')(c8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(c8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Activation('relu')(c8)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Activation('relu')(c9)
    c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(c9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Activation('relu')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def lunet_model_with_batchnorm(input_size=(model_size, model_size, 3)): 
    inputs = layers.Input(input_size)

    # Downsampling
    c1 = layers.Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    c2 = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    c3 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)
    c4 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)
    c5 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)

    # Upsampling
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation('relu')(c6)
    c6 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation('relu')(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Activation('relu')(c7)
    c7 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Activation('relu')(c7)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Activation('relu')(c8)
    c8 = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(c8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Activation('relu')(c8)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Activation('relu')(c9)
    c9 = layers.Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(c9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Activation('relu')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def lunet_model_with_dropoff(input_size=(model_size, model_size, 3)): 
    inputs = layers.Input(input_size)

    # Downsampling
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = layers.Dropout(0.2)(c4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)



    # Bottleneck
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)


    # Upsampling
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model




def unet_with_both(input_size=(model_size, model_size, 3)):
    inputs = layers.Input(input_size)

    # Downsampling
    c1 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.Dropout(0.1)(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    c2 = layers.Dropout(0.1)(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    c3 = layers.Dropout(0.1)(c3)

    # Upsampling
    u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(u1)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)
    c4 = layers.Dropout(0.1)(c4)

    u2 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(u2)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)
    c5 = layers.Dropout(0.1)(c5)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def unet_with_batchnorm(input_size=(model_size, model_size, 3)):
    inputs = layers.Input(input_size)

    # Downsampling
    c1 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)

    # Upsampling
    u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(u1)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)

    u2 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(u2)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def unet_with_dropout(input_size=(model_size, model_size, 3)):
    inputs = layers.Input(input_size)

    # Downsampling
    c1 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Activation('relu')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Activation('relu')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Dropout(0.1)(c3)
    c3 = layers.Activation('relu')(c3)

    # Upsampling
    u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(u1)
    c4 = layers.Dropout(0.1)(c4)
    c4 = layers.Activation('relu')(c4)

    u2 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(u2)
    c5 = layers.Dropout(0.1)(c5)
    c5 = layers.Activation('relu')(c5)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def unet_model(input_size=(model_size, model_size, 3)):
    inputs = layers.Input(input_size)

    # Downsampling
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)

    # Upsampling
    u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u1)
    u2 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u2)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def unet_model_width_preprocess(input_size=(None, None, 3)):
    inputs = layers.Input(input_size)

    # Resize to fixed model size
    x = layers.Resizing(model_size, model_size)(inputs)


    # Downsampling
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)

    # Upsampling
    u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u1)
    u2 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u2)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


#
# Assist functions
#
def load_and_preprocess_image(image_path, mask_path, target_size=model_size):
    #TOGGLE ON FOR TESTING:
    testing = False

    # Load and decode the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)

    # Resize and preserve aspect ratio
    image = tf.image.resize_with_pad(image, target_size, target_size)

    # Normalize the image
    image = (image / 127.5) - 1

    # Load and decode the mask with the same approach
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize_with_pad(mask, target_size, target_size)
    mask = tf.cast(mask, tf.float32) / 255.0

    if(testing == True):
        return image, mask, image_path, mask_path
    else:
        return image, mask

def create_dataset(image_dir, mask_dir, batch_size):
    image_files = os.listdir(image_dir)
    mask_files = set(os.listdir(mask_dir))

    # Ensure pairing by matching filenames, assuming masks have '_mask.png' suffix ie. "IMG_1234_mask.png"
    image_paths = []
    mask_paths = []
    for fname in image_files:
        base_name = os.path.splitext(fname)[0]
        mask_name = base_name + "_pixels0.png"

        if mask_name in mask_files:
            image_paths.append(os.path.join(image_dir, fname))
            mask_paths.append(os.path.join(mask_dir, mask_name))

    # Create a dataset of (image, mask) pairs
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def display_image_and_mask(image, mask, image_name, mask_name):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'Image: {image_name}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask[:, :, 0], cmap='gray')
    plt.title(f'Mask: {mask_name}')
    plt.axis('off')

    plt.show()

def construct_next_path(base_dir="trained_models", prefix="run"):
    os.makedirs(base_dir, exist_ok=True)
    i = 1
    while True:
        model_path = os.path.join(base_dir, f"{prefix}_{i}")
        if not os.path.exists(model_path):
            return model_path
        i += 1

def plot_training_history(history, new_path):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mean_iou = history.history['mean_iou']
    val_mean_iou = history.history['val_mean_iou']
    
    epochs = range(1, len(acc) + 1)

    # Get the latest values
    latest_acc = acc[-1]
    latest_val_acc = val_acc[-1]
    latest_loss = loss[-1]
    latest_val_loss = val_loss[-1]
    latest_iou = mean_iou[-1]
    latest_val_iou = val_mean_iou[-1]
   
    plt.figure(figsize=(15, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title(f'Accuracy\nTrain: {latest_acc:.4f} | Val: {latest_val_acc:.4f}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title(f'Loss\nTrain: {latest_loss:.4f} | Val: {latest_val_loss:.4f}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    
    plt.tight_layout()

    os.makedirs(new_path, exist_ok=True)
    
    history_path = os.path.join(new_path, "fithistory.pkl")
    with open(history_path, 'wb') as file:
        pickle.dump(history.history, file)

    new_path += ("/" + version_name + ".png")
    plt.savefig(new_path, bbox_inches='tight')
    


        

def copy_dir(dst, src):
    print("dst " + str(dst))
    print("src " + str(src))
    if not os.path.exists(dst):
        shutil.copytree(src, dst, symlinks=True, ignore=None)
    else:
        print("Already exists")


#
#  Constants
#

# Create training and validation datasets
train_image_dir = 'dataset/split_images/train'
train_mask_dir = 'dataset/split_images/train_mask'
val_image_dir = 'dataset/split_images/val'
val_mask_dir = 'dataset/split_images/val_mask'
test_image_dir = 'dataset/split_images/test'
test_mask_dir = 'dataset/split_images/test_mask'

#
#   Iteration naming and managing 
#   

ep=32
nick="log_w_test_set"
version_name = nick + "_" +str(model_size) + "_" +str(ep)


model_name = lunet_model_with_both
batch_size = 25
learning_rate=0.001
loss_name='dice'

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=504,
    decay_rate=0.85,
    staircase=False)

optimizer_name=tf.keras.optimizers.Adam(learning_rate=lr_schedule)


#
# Buiilding and running the
#
model = model_name()
model.compile(
    optimizer=optimizer_name,
    loss=loss_name, # string cointaing the loss namne
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.MeanIoU(num_classes=2, name='mean_iou')
    ]
)

model.summary() # Model summary to see the architecture

# creates datasets and runs model
train_dataset = create_dataset(train_image_dir, train_mask_dir, batch_size=batch_size)
val_dataset = create_dataset(val_image_dir, val_mask_dir, batch_size=batch_size)
model_history = model.fit(train_dataset, epochs=ep, validation_data=val_dataset)



# builds path
path = construct_next_path()
# saves model and matadata
plot_training_history(model_history, path)
model.save(path + "/" + version_name + ".keras")
model.save(path + "/" + version_name + ".h5")
model.export(path + "/" + version_name + "_old")

# saves model dataset
os.makedirs(path + "/dataset")
copy_dir(path + "/dataset/val", val_image_dir)
copy_dir(path + "/dataset/val_mask", val_mask_dir)
copy_dir(path + "/dataset/train", train_image_dir)
copy_dir(path + "/dataset/train_mask", train_mask_dir)


# Date and time
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d %H:%M:%S")


# logs metadata to file
log_file = os.path.join(path, "info.txt")
with open(log_file, 'w') as f:
    f.write(f"timestamp = '{timestamp}'\n")
    f.write(f"model_name = {model_name.__name__}\n")
    f.write(f"batch_size = {batch_size}\n")
    f.write(f"learning_rate = {learning_rate}\n")
    f.write(f"optimizer_name='{optimizer_name}'\n")
    f.write(f"loss_name='{loss_name}'\n")
    f.write(f"model_size = {model_size}\n")
    f.write(f"ep = {ep}\n")
    f.write(f"nick = '{nick}'\n")
    f.write(f"version_name = '{version_name}'\n")
    f.write(f"notes = '{notes}'\n")