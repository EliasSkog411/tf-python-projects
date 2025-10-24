# Load the model (optionally)
from tensorflow import keras

import tensorflow
import cv2
import numpy as np
import json
import glob
import os
import tensorflow as tf
from keras import layers, models
import os
from PIL import Image, ExifTags
import os
import matplotlib.pyplot as plt

import sys
from keras.models import load_model

from skimage import measure
import matplotlib.pyplot as plt


import cv2
import numpy as np

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

def load_and_preprocess_image(image_path, mask_path, target_size=256):
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
        image = tf.image.decode_jpeg(image, channels=3)

        if(image[:, :, 0].shape != mask[:, :, 0].shape):
            mask = tf.image.rot90(mask, k=1)
            encoded_mask = tf.io.encode_png(mask)
            tf.io.write_file(mask_paths[i], encoded_mask)
        print("finished " + str(i) + "/" + str(len(mask_paths)))

    exit()

def combine_image(b, foreground):
    background = b.copy()
    print(" ")
    print("background type " + str(type(background)) + " foreground type " + str(type(foreground))) 
    print("background shape " + str(background.shape) + " foreground shape " + str(foreground.shape)) 
    print(" ")
    
    for x in range(256):
        for y in range(256):
            if(foreground[x,y] != 0):
                background[x, y, 0] = 255
                background[x, y, 1] = 0
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


    image = tf.image.resize_with_pad(image, 256, 256)
    image = image.numpy()
    image = image / 255


    mask = tf.image.resize_with_pad(mask, 256, 256)
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


model = load_model("trained_models/run_4/more_validation_50")
test_image_dir = 'Training/Testing-images'
test_mask_dir = 'Training/Testing-masks'

#test_image_dir = 'testc'
#test_mask_dir = 'testd'

# test_image_dir = 'model_validation/input/Selected_images'
# test_mask_dir = 'model_validation/input/Selected_images_annotations'
# test_image_dir = 'model_validation/input/Stock_bilder'
# test_mask_dir = 'model_validation/input/Stock_bilder_annotations'



test_image_dir = 'dataset/split_images/Stock_bilder'
test_mask_dir = 'dataset/split_images/Stock_bilder_mask'



# test_dataset, image_paths, mask_paths = create_dataset(test_image_dir, test_mask_dir, batch_size=1)
# test_image_dir = 'model_validation/merge'
# test_mask_dir = 'model_validation/merge_mask'
test_dataset, image_paths, mask_paths = create_dataset(test_image_dir, test_mask_dir, batch_size=1)



# image_paths, mask_paths = find_images(test_image_dir, test_mask_dir)

# rotate_retarded_set(image_paths, mask_paths)
# exit()

print(len(image_paths))
for i in range(len(image_paths)):
    display_from_path(image_paths[i], mask_paths[i], True)
exit()

i = 0
for images, masks in test_dataset:  
    bugging = False
    if(bugging):
        images = read_image_from_file("test.txt")
        images = (images / 127.5) - 1

    pred_masks_raw = model.predict(images)

    imgnumpy = images
    if(not bugging):
            imgnumpy = imgnumpy.numpy()

    # from PIL import Image
    # imgnumpy = imgnumpy + 1
    # imgnumpy = imgnumpy * 127.5
    # print(images.shape)
    # images = images[0, :, :, :]
    # images = images + 1
    # images = images * 127.5
    # im_rgb = cv2.cvtColor(images.numpy(), cv2.COLOR_BGR2RGB)
    # cv2.imwrite("toswiftRGB.jpg", im_rgb)
    # print(images.shape)
    # exit()
    # for x in range(256): 
    #     for y in range(256):
    #         if(imgnumpy[0,x,y,0] != -1 and imgnumpy[0,x,y,1] != -1 and imgnumpy[0,x,y,2] != -1):
    #             print(imgnumpy[0,x,y,0], end=",")
    #             print(imgnumpy[0,x,y,1], end=",")
    #             print(imgnumpy[0,x,y,2], end="  ")
    #     print("")



    # If the image is in 0-1 range, multiply by 255
    # Load the image using OpenCV
    img_bgr = cv2.imread(image_paths[i])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting
    img_bgr = tf.image.resize_with_pad(img_bgr, 256, 256)
    
    # create an parsable image
    img_scale = tf.image.resize_with_pad(img_rgb, 256, 256)
    img_scale = (img_scale / 255.0)
    img_scale = img_scale.numpy()
    print("imgscale format " + str(img_scale.shape))

    
    img = img_scale

    # Get the raw model prediction (assuming it's a binary mask for simplicity)
    pred_mask = pred_masks_raw[0, :, :, 0]
    pred_mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255
    pred_mask_binary = post_process(pred_mask_binary)
    

    cv2.imwrite(f"mask/mask{i}.png", pred_mask_binary)    


    pog = pred_mask_binary

    # Create a structuring element (kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))

    # First: perform Opening
    opened = cv2.morphologyEx(pog, cv2.MORPH_OPEN, kernel)


    # Label connected components
    labels = measure.label(opened, connectivity=2)
    blobs = measure.regionprops(labels)

    # Find the largest blob (based on area)
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
    cv2.drawContours(contour_image, contours, -1, (255), 1)

    plt.figure(figsize=(20, 5))  # Wider figure for 4 images

    # Original Image + Contours Combined
    plt.subplot(1, 4, 1)
    if(not bugging):
        images = images.numpy()


    images = images[0, :, :, :]
    plt.imshow(img_scale)
    plt.title("Original Image")
    plt.axis('off')

    # Raw Model Prediction
    plt.subplot(1, 4, 2)
    combine = combine_image(img_scale, contour_image)
    plt.imshow(combine, cmap='gray')
    plt.title("Outline")
    plt.axis('off')
    

    # Contour Image
    plt.subplot(1, 4, 3)
    plt.imshow(contour_image, cmap='gray')
    plt.title("Contours")
    plt.axis('off')

    # Largest Blob Mask
    plt.subplot(1, 4, 4)
    plt.imshow(pred_mask, cmap='viridis')
    plt.title("Raw Model Prediction")
    plt.colorbar()
    plt.axis('off')

    plt.suptitle(f"Image: {image_paths[i]}")
    # plt.show()
    plt.savefig(f"benchmark/image_{i}.png", bbox_inches='tight')

    i += 1

