from tensorflow import keras
import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
from PIL import Image, ExifTags
import sys
from keras.models import load_model
from skimage import measure
import matplotlib.pyplot as plt
import pyperclip
import threading
import queue
import time


model_size=512

def make_dir2(path):
    try:
        os.mkdir(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")

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

def get_folders_from_clipboard():
    """
    Reads a newline-separated list of folder paths from the clipboard
    and returns it as a list of strings, stripping any whitespace.
    """
    clipboard_content = pyperclip.paste()
    folder_list = [line.strip() for line in clipboard_content.strip().splitlines() if line.strip()]
    return folder_list



# get val dataset
base_img_dir = "_large_tests"
make_dir2(base_img_dir)

# builds dataset
test_image_dir = 'dataset/split_images/val'
test_mask_dir = 'dataset/split_images/val_mask'
test_dataset, image_paths, mask_paths = create_dataset(test_image_dir, test_mask_dir, batch_size=1)

#builds locks
plt_lock = threading.Lock()
paths_lock = threading.Lock()

#created paths, to be written to stdout (print())
paths = []

def extract_config_subpath(path: str) -> str:
    # Normalize path to handle different OS path separators
    parts = os.path.normpath(path).split(os.sep)
    
    # Return everything after the first two parts
    if len(parts) > 2:
        return os.path.join(*parts[2:])
    else:
        return ''  # Or raise an error if less than 3 parts

def perfomce_inference(model_path):
    # Build folder from model_path
    outputdir = extract_config_subpath(model_path).replace("/", "").replace(".keras", "").replace("_model", "").replace("0,", "0.")
    outputdir = base_img_dir + "/" + outputdir.replace("bs_", "").replace("dr", "").replace("w", "").replace("h", "").replace("ep", "")
    print(f"outputdir: {outputdir}")
    print(f"model_path: {model_path}")

    make_dir2(outputdir)
    model_path = model_path.replace("dr_0,", "dr_0.")
    model = load_model(model_path)


    print(f"dasdasasd {model_path}")
    i = 0
    for images, masks in test_dataset:  
        pred_masks_raw = model.predict(images)

        # Load the image using OpenCV
        img_bgr = cv2.imread(image_paths[i])
        img_scale = tf.image.resize_with_pad(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), model_size, model_size)
        img_scale = (img_scale / 255.0)    

        # Get the raw model prediction (assuming it's a binary mask for simplicity)
        pred_mask = pred_masks_raw[0, :, :, 0]
        pred_mask_binary = (pred_mask > 0.3).astype(np.uint8) * 255

        # Label connected components
        labels, labels_num = measure.label(pred_mask_binary, connectivity=2, return_num=True)
        if(labels_num == 0):
            i = i + 1
            continue

        blobs = measure.regionprops(labels)
        largest_blob = max(blobs, key=lambda b: b.area)
        mask = np.zeros(img_scale.numpy().shape[:2], dtype=np.uint8)  # Optional: Mask only the largest blob
        for coords in largest_blob.coords:
            mask[coords[0], coords[1]] = 255

        # Find contours from the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = np.zeros_like(mask)
        cv2.drawContours(contour_image, contours, -1, (255), 2)

        # Create a new figure
        plt.figure(figsize=(10, 5))  # 2 subplots side by side

        # Normalize and shape the input image
        images = images.numpy()[0]  # shape: (H, W, C)
        images = (images + 1) / 2   # normalize to [0, 1]

        # Subplot 1: Combined Image (Overlay)
        plt.subplot(1, 2, 1)
        combined = combine_image(images, contour_image)
        plt.imshow(combined, cmap='gray')
        plt.title("Outline")
        plt.axis('off')

        # Subplot 2: Prediction Mask
        plt.subplot(1, 2, 2)
        im = plt.imshow(pred_mask, cmap='viridis')
        plt.title("Raw Model Prediction")
        plt.axis('off')        


        # Save to file
        plt.savefig(f"{outputdir}/image_{i}.png", bbox_inches='tight')
        plt.close()  # Optional: close to free memory if looping

        
        i += 1
    

    return outputdir

# Dummy implementation of build_images
def build_images(model_path):
    print(f"[{threading.current_thread().name}] Processing: {model_path}")
    res = perfomce_inference(model_path)
    
    with paths_lock:
        paths.append(res)


    time.sleep(1)  # Simulate work
    print(f"[{threading.current_thread().name}] Done: {model_path}")

# Worker function for threads
def worker(task_queue):
    while True:
        try:
            path = task_queue.get_nowait()
        except queue.Empty:
            break
        try:
            build_images(path)
        finally:
            task_queue.task_done()

# Entry function
def run(paths, threads):
    for path in paths:
        print("apa  " + path)
        build_images(path)

    return


    task_queue = queue.Queue()
    for path in paths:
        task_queue.put(path)

    thread_list = []
    for i in range(threads):
        t = threading.Thread(target=worker, args=(task_queue,), name=f"Thread-{i+1}")
        t.start()
        thread_list.append(t)

    # Wait for all threads to complete
    for t in thread_list:
        t.join()


 # get models
print("Running infernce with models: ")
# raw_models = get_folders_from_clipboard()
raw_models = [line.strip() for line in sys.stdin if line.strip()]
models = []

for i in range(len(raw_models)):
    if(raw_models[i].endswith("_")):
        str = raw_models[i] + "model.keras"
        # models.append(str)
    elif(raw_models[i].endswith("/")):
        str = raw_models[i] + "_model_fin.keras"
        models.append(str)
    print(raw_models[i])


print("\n")
run(models, threads=6)

for pp in paths:
    print(pp)