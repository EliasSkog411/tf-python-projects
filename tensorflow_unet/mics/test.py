import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.callbacks import Callback
import os
import sys

# Get the absolute path to ../train
train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../train'))

# Add it to sys.path
sys.path.append(train_path)

# Now you can import
import unet_models


print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

# model = tf.keras.models.load_model("/home/burk/Documents/model_post_process/model_parse/quant.keras")

# model = unet_models.dynamic_unet_model(input_size=(512, 512, 3), width=2, height=5)
model = unet_models.lunet_model_with_both(input_size=(512, 512, 3))
model.summary()
