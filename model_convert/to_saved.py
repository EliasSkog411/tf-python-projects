import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Permute
from tensorflow.keras.layers import Input, Lambda, Permute

import os

def fix_batch_size(model, fixed_batch_size=1):
    """
    Creates a new model identical to `model` but with a fixed batch size.

    Args:
        model: tf.keras.Model with dynamic batch size (None)
        fixed_batch_size: int, batch size to fix (e.g. 1)

    Returns:
        new_model: tf.keras.Model with fixed batch size in input/output specs
    """

    # Get original input shape excluding batch dimension
    original_input_shape = model.input_shape[1:]  # e.g., (1, 512, 512)

    # Create new Input with fixed batch size
    new_input = Input(shape=original_input_shape, batch_size=fixed_batch_size)

    # Run the original model on new input
    new_output = model(new_input)

    # Build new model
    new_model = Model(inputs=new_input, outputs=new_output)
    return new_model


def change_output_format(model):
    """
    Modifies a Keras model to change its output shape from
    (None, H, W, C) to (None, C, H, W) using a Permute layer.

    Parameters:
        model (tf.keras.Model): The original model.

    Returns:
        tf.keras.Model: A new model with permuted output shape.
    """
    # Get the original model input
    inputs = model.input

    # Get the original output
    original_output = model.output

    # Apply Permute to change from (H, W, C) to (C, H, W)
    permuted_output = Permute((3, 1, 2))(original_output)

    # Create new model with permuted output
    new_model = Model(inputs=inputs, outputs=permuted_output)
    return new_model

def fix_batch_and_output_shape(model):
    """
    Creates a new model with:
      - Fixed batch size = 1
      - Output shape changed from (1, H, W, C) to (1, C, H, W)

    Assumes original model output shape is (None, H, W, C)
    """
    # Get original input shape (excluding batch)
    input_shape = model.input_shape[1:]  # e.g., (1, 512, 512)

    # Create input with fixed batch size
    new_input = Input(shape=input_shape, batch_size=1)

    # Get output from original model
    output = model(new_input)

    # Permute to reorder output from (1, 512, 512, 1) to (1, 1, 512, 512)
    permuted_output = Permute((4, 2, 3))(tf.expand_dims(output, axis=0)) if len(output.shape) == 3 else Permute((3, 1, 2))(output)

    # Create new model
    new_model = Model(inputs=new_input, outputs=permuted_output)

    return new_model

def fix_batch_and_output_shape2(model):
    """
    Wraps a Keras model to:
      - Fix batch size to 1
      - Convert RGB to grayscale if needed
      - Rescale output from [0, 1] → [0, 255]
      - Reorder output shape to (1, 1, H, W)
    """

    # Input shape without batch dimension
    input_shape = model.input_shape[1:]  # e.g., (256, 256, 3)
    new_input = Input(shape=input_shape, batch_size=1, name="keras_tensor")

    # Run through original model
    output = model(new_input)  # shape: (1, H, W, C)

    # Convert to grayscale if output is RGB
    if output.shape[-1] == 3:
        output = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(output)  # (1, H, W, 1)

    # Rescale from [0, 1] to [0, 255]
    output = Lambda(lambda x: tf.clip_by_value(x * 255.0, 0.0, 255.0))(output)

    # Reorder to (1, 1, H, W)
    output = Permute((3, 1, 2))(output)  # Keras Permute uses 1-based indexing

    return Model(inputs=new_input, outputs=output, name="fixed_model")



# --- Configuration ---
#keras_model_path = "tensorflow_models/temp_fast_cs.keras"       # Path to the .keras file

keras_model_path = "tensorflow_models/very_good_cs.keras"


#saved_model_dir = "tensorflow_models/ep_48_model_savemodel_permute"    # Output directory for SavedModel
#saved_model_dir = "tensorflow_models/final_savemodel_permute"    # Output directory for SavedModel
#saved_model_dir = "tensorflow_models/temp_fast_cs_permute"    # Output directory for SavedModel

saved_model_dir = "tensorflow_models/VERY_GOOD_CS_PERMUTE"

new_input_name = "keras_tensor"        #

# --- Load the .keras model ---


print(f"Loading model from: {keras_model_path}")
model = tf.keras.models.load_model(keras_model_path)
model = fix_batch_and_output_shape2(model)

# --- Print model summary ---
print("\n📋 Model Summary:")
model.summary()

# --- Save as TensorFlow SavedModel ---
print(f"\nSaving model to: {saved_model_dir}")

model.export(saved_model_dir)

print("✅ Conversion complete.")