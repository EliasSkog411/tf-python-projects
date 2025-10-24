import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import numpy as np



ep=50
nick="more_validation"
version_name = nick + "_" +str(ep)

def unet_model(input_size=(256, 256, 3)):
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

def load_and_preprocess_image(image_path, mask_path, target_size=256):
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

# Giving a visual check that everything looks as expected
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

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    new_path = construct_next_path(base_dir="trained_models_info")

    os.makedirs(new_path)
    new_path += ("/" + version_name + ".png")
    plt.savefig(new_path, bbox_inches='tight')

#    plt.show()


# Create the U-Net model
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary() # Model summary to see the architecture

# Create training and validation datasets
train_image_dir = 'testset/Train-images'
train_mask_dir = 'testset/Train-masks'
val_image_dir = 'testset/Validation-images'
val_mask_dir = 'testset/Validation-masks'
test_image_dir = 'testset/Testing-images'
test_mask_dir = 'testset/Testing-masks'


train_image_dir = 'dataset/split_images/train'
train_mask_dir = 'dataset/split_images/train_mask'
val_image_dir = 'dataset/split_images/val'
val_mask_dir = 'dataset/split_images/val_mask'
test_image_dir = 'dataset/split_images/test'
test_mask_dir = 'dataset/split_images/test_mask'


test_dataset = create_dataset(test_image_dir, test_mask_dir, batch_size=32)
train_dataset = create_dataset(train_image_dir, train_mask_dir, batch_size=32)
val_dataset = create_dataset(val_image_dir, val_mask_dir, batch_size=32)

h = model.fit(train_dataset, epochs=ep, validation_data=val_dataset)

# Call the function with your training history
plot_training_history(h)




path = construct_next_path()
os.makedirs(path)

# epoches _ (train,validation,test)
model.save(path + "/" + version_name)
# tf.saved_model.save(
#     model,
#     path + ".keras",
#     options=tf.saved_model.SaveOptions(experimental_custom_gradients=False)
# )

# model.export(path + "/old")
#50 - 0.9588
#100 - 0.9658
#200 0.9727
