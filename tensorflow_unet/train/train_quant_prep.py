import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"
import tensorflow as tf
import tf_keras as keras
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
import shutil
import pickle



import unet_models

notes = input("Enter any notes for this run: ") 
model_size = 512


# #'
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

    plt.figure(figsize=(18, 5))

    # Accuracy plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title(f'Accuracy\nTrain: {latest_acc:.4f} | Val: {latest_val_acc:.4f}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title(f'Loss\nTrain: {latest_loss:.4f} | Val: {latest_val_loss:.4f}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Mean IoU plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, mean_iou, 'bo-', label='Training Mean IoU')
    plt.plot(epochs, val_mean_iou, 'ro-', label='Validation Mean IoU')
    plt.title(f'Mean IoU\nTrain: {latest_iou:.4f} | Val: {latest_val_iou:.4f}')
    plt.xlabel('Epochs')
    plt.ylabel('Mean IoU')
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

ep=1
nick="reee_first_q_prep"
version_name = nick + "_" +str(model_size) + "_" +str(ep)


model_name = unet_models.dynamic_unet_model
batch_size = 16
learning_rate=0.001
loss_name='binary_crossentropy'

optimizer_name = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# optimizer_name = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

quantize_model = tfmot.quantization.keras.quantize_model

#
# Buiilding and running the
#
model = model_name(width=2, height=5, input_size=(model_size, model_size, 3)) # model_name(input_size=(model_size, model_size, 3), width=3, height = 6)
model.compile(
    optimizer=optimizer_name,
    loss=loss_name, # string cointaing the loss namne
    metrics=[
        tf.keras.metrics.MeanIoU(num_classes=2, name='mean_iou'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    ]
)

model.summary() # Model summary to see the architecture


# creates datasets and runs model
train_dataset = create_dataset(train_image_dir, train_mask_dir, batch_size=batch_size)
val_dataset = create_dataset(val_image_dir, val_mask_dir, batch_size=batch_size)
model_history = model.fit(train_dataset, epochs=ep, validation_data=val_dataset)
tf.keras.backend.clear_session()
tf.keras.backend.clear_session()
tf.keras.backend.clear_session()


# builds path
path = construct_next_path()
# saves model and matadata


plot_training_history(model_history, path)
model.save(path + "/" + version_name + ".keras")
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

print("finished")