import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import shutil
import pickle


def conv_block(last_layer, filters, dropout_rate):
    layer = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(last_layer)
    layer = layers.BatchNormalization()(layer)
    layer = layers.Activation('relu')(layer)
    
    if(dropout_rate != 0): 
        layer = layers.Dropout(dropout_rate)(layer)

    return layer

def downsampling_block(last_layer, width, filters, dropout_rate=0.1):
    layer = last_layer

    # Width of model
    for _ in range(width - 1):
        layer = conv_block(layer, filters, dropout_rate)
    layer = conv_block(layer, filters, 0)

    # downscaling for next layer
    downscaled = layers.MaxPooling2D((2, 2))(layer)       
    downscaled = layers.Dropout(dropout_rate)(downscaled)

    return layer, downscaled

def upsampling_block2(last_layer, width, filters, dropout_rate, across):
    layer = layers.UpSampling2D()(last_layer)
    layer = conv_block(layer, filters, dropout_rate)
    layer = layers.concatenate([layer, across])
    
    for _ in range(width):
        layer = conv_block(layer, filters, dropout_rate)
 
    return layer

def upsampling_block(last_layer, width, filters, dropout_rate, across):
    layer = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(last_layer)
    layer = layers.concatenate([layer, across])
    
    for _ in range(width):
        layer = conv_block(layer, filters, dropout_rate)
 
    return layer
    
def dynamic_unet_model(input_size=(256, 256, 3), width=2, height=4, dropout_rate=0.2, start_filter=16, accelerations=2):
    inputs = layers.Input(input_size)
    last_layer = inputs
    filters = start_filter
    skip_connections = [] 

    # Downsampling path
    for i in range(height - 1):
        temp_layer, downscaled = downsampling_block(last_layer, width, filters, dropout_rate)
        skip_connections.append(temp_layer)
        last_layer = downscaled
        filters *= accelerations

    # Bottleneck
    for _ in range(width):
        last_layer = conv_block(last_layer, filters, 0.3)
 
    # Upsampling path
    for skip in reversed(skip_connections):
        filters //= accelerations
        last_layer = upsampling_block(last_layer, width, filters, dropout_rate, skip)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(last_layer)
    model = models.Model(inputs=[inputs], outputs=[outputs])

    return model


#
#   Models 1554 / 15
#
def lunet_model_with_both(input_size=(256, 256, 3)): 
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

def lunet_model_with_batchnorm(input_size=(256, 256, 3)): 
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

def lunet_model_with_dropoff(input_size=(256, 256, 3)): 
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




def unet_with_both(input_size=(256, 256, 3)):
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

def unet_with_batchnorm(input_size=(256, 256, 3)):
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

def unet_with_dropout(input_size=(256, 256, 3)):
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

def b(input_size=(512, 512, 3)):
    inputs = layers.Input(input_size)

    # Downsampling
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
 
    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c1)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def a(input_size=(512, 512, 3)):
    inputs = layers.Input(input_size)

    # Downsampling
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    # p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    # p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
 



    # Upsampling
    # u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c3)
    # u1 = layers.concatenate([u1, c2])
    # c4 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u1)
    # u2 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c4)
    # u2 = layers.concatenate([u2, c1])
    # c5 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u2)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c3)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


# def conv_block(x, filters, dropout_rate=0.1):
#     x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     x = layers.Dropout(dropout_rate)(x, training=True)  # MC Dropout
#     x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     x = layers.Dropout(dropout_rate)(x, training=True)  # MC Dropout
#     return x

# def bayesian_unet(input_shape=(256, 256, 3), dropout_rate=0.1):
#     inputs = layers.Input(shape=input_shape)

#     # Encoder
#     c1 = conv_block(inputs, 16, dropout_rate)
#     p1 = layers.MaxPooling2D((2, 2))(c1)

#     c2 = conv_block(p1, 32, dropout_rate)
#     p2 = layers.MaxPooling2D((2, 2))(c2)

#     c3 = conv_block(p2, 64, dropout_rate)
#     p3 = layers.MaxPooling2D((2, 2))(c3)

#     c4 = conv_block(p3, 128, dropout_rate)
#     p4 = layers.MaxPooling2D((2, 2))(c4)

#     # Bottleneck
#     c5 = conv_block(p4, 256, dropout_rate)

#     # Decoder
#     u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
#     u6 = layers.concatenate([u6, c4])
#     c6 = conv_block(u6, 128, dropout_rate)

#     u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
#     u7 = layers.concatenate([u7, c3])
#     c7 = conv_block(u7, 64, dropout_rate)

#     u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
#     u8 = layers.concatenate([u8, c2])
#     c8 = conv_block(u8, 32, dropout_rate)

#     u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
#     u9 = layers.concatenate([u9, c1])
#     c9 = conv_block(u9, 16, dropout_rate)

#     outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

#     model = models.Model(inputs=[inputs], outputs=[outputs])
#     return model
