import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.callbacks import Callback


import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
import shutil
import pickle
import gc

import unet_models
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


model_size = 512

# Create training and validation datasets
train_image_dir = 'dataset/split_images/train'
train_mask_dir = 'dataset/split_images/train_mask'
val_image_dir = 'dataset/split_images/val'
val_mask_dir = 'dataset/split_images/val_mask'
test_image_dir = 'dataset/split_images/test'
test_mask_dir = 'dataset/split_images/test_mask'

def make_dir2(path):
    try:
        os.mkdir(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")

def construct_next_path(base_dir="512_dynamic_unet", prefix="run"):
    make_dir2(base_dir)
    i = 1
    while True:
        model_path = os.path.join(base_dir, f"{prefix}_{i}")
        if not os.path.exists(model_path):
            make_dir2(model_path)            
            return model_path
        i += 1

base_dir = "512_dynamic_unet/run_7_combined" #"512_dynamic_unet/run_1"
make_dir2(base_dir)
running_info = base_dir + "/running_info.txt"

if not os.path.exists(running_info):
    with open(running_info, 'ab') as f:
        line = f"name_id\tlatest_acc\tlatest_loss\tlatest_iou\tlatest_val_acc\tlatest_val_loss\tlatest_val_iou\n"
        f.write(line.encode('utf-8'))  # or 'ascii' if applicable
        # f.write(f"latest_acc\tlatest_loss\tlatest_iou\tlatest_val_acc\tlatest_val_loss\tlatest_val_iou\n")
 

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

def plot_training_history(history, new_path, plot_name="train_plot.png"):
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

    history_path = new_path +"fh.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)

    with open(running_info, 'a') as f:
        f.write(f"{new_path}\t{latest_acc:.4f}\t{latest_loss:.4f}\t{latest_iou:.4f}\t{latest_val_acc:.4f}\t{latest_val_loss:.4f}\t{latest_val_iou:.4f}\n".replace(".", ","))

    new_path += plot_name
    plt.savefig(new_path, bbox_inches='tight')
    plt.close()
    
def copy_dir(dst, src):
    print("dst " + str(dst))
    print("src " + str(src))
    if not os.path.exists(dst):
        shutil.copytree(src, dst, symlinks=True, ignore=None)
    else:
        print("Already exists")


# model
# |   batch_size
# |     | 0.6
# |     | 0.7
# |     | 0.8
# |     | 0.9
class CustomCheckpointCallback(Callback):
    def __init__(self, save_every_n_epochs, path):
        super().__init__()
        self.save_every_n_epochs = save_every_n_epochs
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        print("current epoch: " + str(epoch + 1))
        if (epoch + 1) % self.save_every_n_epochs == 0:
            path = self.path
            path += "/ep_" + str(epoch + 1) + "_"
            # Save training history plot
            plot_training_history(self.model.history, path)
            # Save model in multiple formats
            self.model.save(path + "model.keras")

# def run_unet_test(widths, heighs, epoch, batch_size):
#     current = 0
#     for w in widths:
#         tf.keras.backend.clear_session()
#         gc.collect()       
#                 # make dir for model
#         directory_name = base_dir + "/w_" + str(w)
#         make_dir2(directory_name)

#         for h in heighs:
#             tf.keras.backend.clear_session()
#             gc.collect()
            
#             # make dir for batch_size
#             directory_name = base_dir + "/w_" + str(w) + "/h_" + str(h)
#             make_dir2(directory_name)

#             # checks if model already exists
#             test_name = directory_name + "/_fin.png"              
#             print(test_name)
#             if os.path.exists(test_name):                
#                 print("Found file: " + test_name  + ". Skipping...")
#                 continue
            
#             make_dir2(directory_name)
#             print(f"\n--- Training  | Width: {w} | Height: {h} ---")
            
#             # Define learning rate schedule
#             # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#             #     initial_learning_rate=0.001,
#             #     decay_steps=504 * (25 / batch_size), # 8 epoches, 63 per step
#             #     decay_rate=decay_rate,
#             #     staircase=False
#             # )
            
#             # Create and compile model
#             model = unet_models.dynamic_unet_model(input_size=(512, 512, 3), width=w, height=h)
#             model.compile(
#                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                 loss='dice', 
#                 metrics=[
#                     tf.keras.metrics.MeanIoU(num_classes=2, name='mean_iou'),
#                     tf.keras.metrics.BinaryAccuracy(name='accuracy'),
#                 ]
#             )
            
#             # Create callback
#             checkpoint_cb = CustomCheckpointCallback(
#                 save_every_n_epochs=8,
#                 path=directory_name
#             )
            
#             # Prepare dataset with batch size
#             train_dataset = create_dataset(train_image_dir, train_mask_dir, batch_size=batch_size)
#             val_dataset = create_dataset(val_image_dir, val_mask_dir, batch_size=batch_size)
            
#             # Train model
#             model_history = model.fit(
#                 train_dataset,
#                 epochs=epoch,
#                 validation_data=val_dataset,
#                 callbacks=[checkpoint_cb]
#             )
            
            
#             directory_name += "/" 
#             plot_training_history(model_history, directory_name, "_plot_fin.png")
#             model.save(directory_name + "_model_fin.keras")
#             current += 1
#             print("finished " + str(current))
            
#             del model
#             tf.keras.backend.clear_session()
                       
                
#             gc.collect()

def test_diffrent_models_bs_dc(batch_sizes, decay_rates, what_the_num, widths, heighs, epoch):
    current = 0
    for bs in batch_sizes:
        tf.keras.backend.clear_session()
        gc.collect()
        
        # make dir for model
        directory_name = base_dir + "/bs_" + str(bs)
        make_dir2(directory_name)

        for dr in decay_rates:
            tf.keras.backend.clear_session()
            gc.collect()
            
            # make dir for model
            directory_name = base_dir + "/bs_" + str(bs) + "/dr_" + str(dr)
            make_dir2(directory_name)

            for num in what_the_num:
                tf.keras.backend.clear_session()
                gc.collect()
                
                # make dir for model
                directory_name = base_dir + "/bs_" + str(bs) + "/dr_" + str(dr) + "/n_" + str(num)
                make_dir2(directory_name)

                for w in widths:
                    tf.keras.backend.clear_session()
                    gc.collect()
                    
                    # make dir for model
                    directory_name = base_dir + "/bs_" + str(bs) + "/dr_" + str(dr) + "/n_" + str(num) + "/w_" + str(w)
                    make_dir2(directory_name)

                    for h in heighs:
                        tf.keras.backend.clear_session()
                        gc.collect()
                        
                        # make dir for batch_size
                        directory_name = base_dir + "/bs_" + str(bs) + "/dr_" + str(dr) + "/n_" + str(num) + "/w_" + str(w) +  "/h_" + str(h)
                        make_dir2(directory_name)

                        # checks if model already exists
                        test_name = directory_name + "/_fin.png"              
                        print(test_name)
                        if os.path.exists(test_name):                
                            print("Found file: " + test_name  + ". Skipping...")
                            continue
                        
                        make_dir2(directory_name)
                        print(f"\n--- Training  | Batch size: {bs} | Decay rate: {dr} | Width: {w} | Height: {h} ---")
                        
                        # Define learning rate schedule
                        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate=0.001,
                            decay_steps=num, # 8 epoches, 63 per step
                            decay_rate=dr,
                            staircase=False
                        )
                        
                        # Create and compile model
                        model = unet_models.dynamic_unet_model(input_size=(512, 512, 3), width=w, height=h)
                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                            loss='binary_crossentropy', 
                            metrics=[
                                tf.keras.metrics.MeanIoU(num_classes=1, name='mean_iou'),
                                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                            ]
                        )
                        
                        # Create callback
                        checkpoint_cb = CustomCheckpointCallback(
                            save_every_n_epochs=8,
                            path=directory_name
                        )
                        
                        # Prepare dataset with batch size
                        train_dataset = create_dataset(train_image_dir, train_mask_dir, batch_size=bs)
                        val_dataset = create_dataset(val_image_dir, val_mask_dir, batch_size=bs)
                        
                        # Train model
                        model_history = model.fit(
                            train_dataset,
                            epochs=epoch,
                            validation_data=val_dataset,
                            callbacks=[checkpoint_cb]
                        )
                        
                        
                        directory_name += "/" 
                        plot_training_history(model_history, directory_name, "_fin.png")
                        model.save(directory_name + "_model_fin.keras")

                        current += 1
                        print("finished " + str(current))
                        
                        del model
                        tf.keras.backend.clear_session()
                                
                            
                        gc.collect()

def test_diffrent_filters_one_model(width=3, height=6, starts=[16], accelerations=[2], epoch=48):
    current = 0
    for srt in starts:
        tf.keras.backend.clear_session()
        gc.collect()
        
        # make dir for model
        directory_name = base_dir + "/srt_" + str(srt)
        make_dir2(directory_name)
        for acc in accelerations:
            tf.keras.backend.clear_session()
            gc.collect()
            
            # make dir for model
            directory_name = base_dir + "/srt_" + str(srt) + "/acc_" + str(acc)
            make_dir2(directory_name)

            # checks if model already exists
            test_name = directory_name + "/_fin.png"              
            print(test_name)
            if os.path.exists(test_name):                
                print("Found file: " + test_name  + ". Skipping...")
                continue

            make_dir2(directory_name)
            print(f"\n--- Training  | Filter start size: {srt} | Filter acceleration: {acc}  ---")

            # Define learning rate schedule
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=504 * (25 / 12), # 8 epoches, 63 per step
                decay_rate=0.9,
                staircase=False
            )

            # Create and compile model
            model = unet_models.dynamic_unet_model(input_size=(512, 512, 3), width=width, height=height, start_filter=srt, accelerations=acc)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy', 
                metrics=[
                    tf.keras.metrics.MeanIoU(num_classes=1, name='mean_iou'),
                    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                ]
            )

            # Create callback
            checkpoint_cb = CustomCheckpointCallback(
                save_every_n_epochs=8,
                path=directory_name
            )

            # Prepare dataset with batch size
            train_dataset = create_dataset(train_image_dir, train_mask_dir, batch_size=12)
            val_dataset = create_dataset(val_image_dir, val_mask_dir, batch_size=12)

            # Train model
            model_history = model.fit(
                train_dataset,
                epochs=epoch,
                validation_data=val_dataset,
                callbacks=[checkpoint_cb]
            )


            directory_name += "/" 
            plot_training_history(model_history, directory_name, "_fin.png")
            model.save(directory_name + "_model_fin.keras")

            current += 1
            print("finished " + str(current))

            del model
            tf.keras.backend.clear_session()
                    
            gc.collect()

        
            


tf.keras.backend.clear_session()

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# epoches managing
epochs = 48

# iteration managing
# model_names = [lunet_model_with_batchnorm, lunet_model_with_both, lunet_model_with_dropoff, bayesian_unet]
# batch_sizes = [8,12,16,20,25]
# decay_rates = [0.70,0.75,0.8,0.85,0.9,0.95]
# run_all_experiments(model_names, batch_sizes, decay_rates, epochs)htop
# run_unet_test([2], [2], epochs, 24)

batch_sizes = [8,12]
decay_rates = [0.9, 0.95]
wat_num = [5 * 66, 10 * 66, 15 * 66, 20 * 66, 30 * 66]
widths = [3]
heights = [6]
test_diffrent_models_bs_dc(batch_sizes, decay_rates, wat_num, widths, heights, epochs)

# starts=[12, 14, 16, 18, 20]
# accs=[1, 2, 3]
# test_diffrent_filters_one_model(starts=starts, accelerations=accs)


make_dir2(base_dir + "/dataset")
copy_dir(base_dir + "/dataset/val", val_image_dir)
copy_dir(base_dir + "/dataset/val_mask", val_mask_dir)
copy_dir(base_dir + "/dataset/train", train_image_dir)
copy_dir(base_dir + "/dataset/train_mask", train_mask_dir)
print("first model complete")

train_image_dir = 'dataset/split_images2/train'
train_mask_dir = 'dataset/split_images2/train_mask'
val_image_dir = 'dataset/split_images2/val'
val_mask_dir = 'dataset/split_images2/val_mask'

base_dir = "512_dynamic_unet/run_7_middle_only"
make_dir2(base_dir)
running_info = base_dir + "/running_info.txt"
tf.keras.backend.clear_session()
test_diffrent_models_bs_dc(batch_sizes, decay_rates, widths, heights, epochs)


print("finished")



