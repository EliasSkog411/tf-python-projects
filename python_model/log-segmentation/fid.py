# Load the model (optionally)
from tensorflow import keras
import os
import tensorflow
import cv2
import numpy as np
import json
import glob
import tensorflow as tf
from keras import layers, models
from PIL import Image, ExifTags
import sys
from keras.models import load_model
from skimage import measure
import matplotlib.pyplot as plt


model_size=512

def read_image_from_file(filepath):
    image_data = []

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            row = []
            for pixel in line.split():
                try:
                    r, g, b = map(int, pixel.split(','))
                    row.append([r, g, b])
                except ValueError:
                    raise ValueError(f"Invalid pixel format: '{pixel}'")
            image_data.append(row)

    image = np.array(image_data, dtype=np.uint8)

    # Add batch dimension: shape becomes (1, height, width, 3)
    return np.expand_dims(image, axis=0)

def create_dataset(image_dir, mask_dir, batch_size):
    image_files = os.listdir(image_dir)
    mask_files = set(os.listdir(mask_dir))

    # Ensure pairing by matching filenames, assuming masks have '_mask.png' suffix ie. "IMG_1234_mask.png"
    image_paths = []
    mask_paths = []
    for fname in image_files:
        base_name = os.path.splitext(fname)[0]
        mask_name = base_name + "_pixels0.png"
        # mask_name = base_name + "_mask.png"

        if mask_name in mask_files:
            image_paths.append(os.path.join(image_dir, fname))
            mask_paths.append(os.path.join(mask_dir, mask_name))

    print(len(image_paths), len(mask_paths))  # Should be equal
    print(image_paths[:3], mask_paths[:3])

    # Create a dataset of (image, mask) pairs
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset, image_paths, mask_paths

def load_and_preprocess_image(image_path, mask_path, target_size=model_size):
    #TOGGLE ON FOR zTESTING:
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

def rotate_retarded_set(image_paths, mask_paths): 
    for i in range(len(mask_paths)):
        mask = tf.io.read_file(mask_paths[i])
        mask = tf.image.decode_jpeg(mask, channels=1)

        image = tf.io.read_file(image_paths[i])
        image = tf.image.decode_jpeg(image, channels=1)

        if(image[:, :, 0].shape != mask[:, :, 0].shape):
            print("FOUND NON MATCHING AT " + str(i))
            mask = tf.image.rot90(mask, k=1)
            encoded_mask = tf.io.encode_png(mask)
            tf.io.write_file(mask_paths[i], encoded_mask)
        print("finished " + str(i + 1) + "/" + str(len(mask_paths)))

    exit()

def combine_image(b, foreground):
    background = b.copy()
    # print("background shape " + str(background.shape) + " foreground shape " + str(foreground.shape)) 
    
    for x in range(model_size):
        for y in range(model_size):
            if(foreground[x,y] != 0):
                background[x, y, 0] = 0.66666666666
                background[x, y, 1] = 1
                background[x, y, 2] = 0
            
    
    # Find where grayscale image is white
    #mask = foreground == 255

    # Set those pixels in HSV image to red (0, 255, 255) — red in HSV
    #background[mask] = [0, 255, 255]

    return background

def find_images(image_dir, mask_dir):
    image_files = os.listdir(image_dir)
    mask_files = set(os.listdir(mask_dir))

    # Ensure pairing by matching filenames, assuming masks have '_mask.png' suffix ie. "IMG_1234_mask.png"
    image_paths = []
    mask_paths = []
    for fname in image_files:
        base_name = os.path.splitext(fname)[0]
        mask_name = base_name + "_pixels0.png"
        # mask_name = base_name + "_mask.png"
        print(base_name)

        if mask_name in mask_files:
            image_paths.append(os.path.join(image_dir, fname))
            mask_paths.append(os.path.join(mask_dir, mask_name))

    return image_paths, mask_paths

def display_from_path(image_path, mask_path, save_instead):
    # loads the images
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)


    image = tf.image.resize_with_pad(image, model_size, model_size)
    image = image.numpy()
    image = image / 255


    mask = tf.image.resize_with_pad(mask, model_size, model_size)
    mask = mask.numpy()
    mask = mask / 255

    combine = combine_image(image, mask)


    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'Image: {image_path}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(combine)
    plt.title(f'Mask: {mask_path}')
    plt.axis('off')

    if(save_instead):
        plt.savefig(f"benchmark/image_{i}.png", bbox_inches='tight')
    else:
        plt.show()

def post_process(mask):
    # Apply morphological operations, e.g., opening to remove noise
    kernel = np.ones((10,10),np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opening
    

model = load_model("log.keras")


#test_image_dir = 'testc'
#test_mask_dir = 'testd'



test_image_dir = 'test'
test_mask_dir = 'test_annotations'



test_dataset, image_paths, mask_paths = create_dataset(test_image_dir, test_mask_dir, batch_size=1)

# test_image_dir = 'dataset/raw_images/LogendsMay2025'
# test_mask_dir = 'dataset/raw_images/LogendsMay2025_mask'
# image_paths, mask_paths = find_images(test_image_dir, test_mask_dir)

# # rotate_retarded_set(image_paths, mask_paths)
# # exit()

# print(len(mask_paths))
# for i in range(len(image_paths)):
#     display_from_path(image_paths[i], mask_paths[i], True)
#     print("finished " + str(i + 1) + "/" + str(len(image_paths)))

# exit()

print("1")
i = 0
for images, masks in test_dataset:  
    print(" ")

    bugging = False
    if(bugging):
        images = read_image_from_file("test.txt")
        images = (images / 127.5) - 1

    pred_masks_raw = model.predict(images)

    imgnumpy = images
    if(not bugging):
            imgnumpy = imgnumpy.numpy()


    # If the image is in 0-1 range, multiply by 255
    # Load the image using OpenCV
    img_bgr = cv2.imread(image_paths[i])
    img_scale = tf.image.resize_with_pad(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), model_size, model_size)
    img_scale = (img_scale / 255.0)    
    img = img_scale.numpy()

    # Get the raw model prediction (assuming it's a binary mask for simplicity)
    pred_mask = pred_masks_raw[0, :, :, 0]
    pred_mask_binary = (pred_mask > 0.3).astype(np.uint8) * 255

    # Create a structuring element (kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    opened = cv2.morphologyEx(pred_mask_binary, cv2.MORPH_OPEN, kernel)


    # Label connected components
    labels, labels_num = measure.label(opened, connectivity=2, return_num=True)
    if(labels_num == 0):
        i = i + 1
        continue
    
    blobs = measure.regionprops(labels)
    largest_blob = max(blobs, key=lambda b: b.area)
    # Extract the largest blob
    minr, minc, maxr, maxc = largest_blob.bbox
    largest_blob_img = img[minr:maxr, minc:maxc]
    # Optional: Mask only the largest blob
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
     
    for coords in largest_blob.coords:
        mask[coords[0], coords[1]] = 255


    # Find contours from the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create a blank image for drawing contours
    contour_image = np.zeros_like(mask)
    # Draw all contours on the image (white color, thickness 2)
    cv2.drawContours(contour_image, contours, -1, (255), 2)
    
    

    # Create a new figure FIRST with appropriate size
    plt.figure(figsize=(20, 5))  # 4 images horizontally

    # Ensure `images` is in correct range and shape
    images = images.numpy()[0]  # shape: (H, W, C)
    images = (images + 1) / 2   # normalize to [0, 1]

    # Subplot 1: Original Image
    plt.subplot(1, 4, 1)
    plt.imshow(images)
    plt.title("Original Image")
    plt.axis('off')

    # Subplot 2: Combined Image (Overlay)
    plt.subplot(1, 4, 2)
    combined = combine_image(images, contour_image)
    plt.imshow(combined, cmap='gray')
    plt.title("Outline")
    plt.axis('off')

    # Subplot 3: Contour
    plt.subplot(1, 4, 3)
    plt.imshow(contour_image, cmap='gray')
    plt.title("Contours")
    plt.axis('off')

    # Subplot 4: Prediction Mask (with optional colorbar)
    ax = plt.subplot(1, 4, 4)
    plt.imshow(pred_mask, cmap='viridis')
    plt.title("Raw Model Prediction")
    plt.colorbar()
    plt.axis('off')
    
    # Main title
    plt.suptitle(f"Image: {image_paths[i]}")

    # Save to file
    plt.savefig(f"benchmark/image_{i}.png", bbox_inches='tight')
    plt.close()  # Optional: close to free memory if looping

    print("finished " + str(i + 1) + "/" + str(len(test_dataset)))
    
    i += 1

