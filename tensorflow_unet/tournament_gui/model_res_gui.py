import os
import numpy as np
from PIL import Image
import dearpygui.dearpygui as dpg
import sys

# --- Config ---
img_width = 794
img_height = 394
img_scale = 1
window_width = 1890
window_height = 950
SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
folders = []

current_index = 1
current_winner = "folders[0]"



first_round = True


# left_tags = []
# right_tags = []
# def clear_tags():
#     for tag in left_tags:

def get_image_files(folder_path):
    return sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(SUPPORTED_FORMATS)
    ])

def load_image(path):
    try:
        img = Image.open(path).convert("RGBA")
        img.thumbnail((img_width, img_height), Image.LANCZOS)
        width, height = img.size
        image_data = np.array(img) / 255.0
        flat_data = image_data.flatten().astype(np.float32)
        return width, height, flat_data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None, None

def write_result(preferred_folder):
    with open("res.txt", "a") as f:
        f.write(f"{preferred_folder}\n")
    print(f"[INFO] Final Winner: {preferred_folder}")


def mark_left_as_better(sender, app_data, user_data):
    global current_winner
    current_winner = user_data["left"]
    
    with open("res.txt", "a") as f:
        f.write(f"{current_winner}\n")


    next_round()

def mark_right_as_better(sender, app_data, user_data):
    global current_winner
    current_winner = user_data["right"]

    with open("res.txt", "a") as f:
        f.write(f"{current_winner}\n")

    next_round()

def create_gui(left_folder, right_folder):
    left_images = get_image_files(left_folder)
    right_images = get_image_files(right_folder)

    min_length = min(len(left_images), len(right_images))
    left_images = left_images[:min_length]
    right_images = right_images[:min_length]


    print(f"LEFT folder {left_folder} RIGHT folder {right_folder}")
    with dpg.texture_registry(show=False):
        for idx, (l_path, r_path) in enumerate(zip(left_images, right_images)):
            left_id = f"left_folder_{idx}"
            right_id = f"right_folder_{idx}"


            lw, lh, ldata = load_image(l_path)
            rw, rh, rdata = load_image(r_path)

            if lw is None or rw is None:
                print(f"Skipping index {idx} due to failed image load")
                continue  # skip broken images

            try:
                if not dpg.does_item_exist(left_id):
                    dpg.add_dynamic_texture(lw, lh, ldata, tag=left_id)
                else:
                    dpg.set_value(left_id, ldata)
            except Exception as e:
                print(f"Failed to add left texture {idx}: {e}")



            try:
                if not dpg.does_item_exist(right_id):
                    dpg.add_dynamic_texture(rw, rh, rdata, tag=right_id)
                else:
                    dpg.set_value(right_id, rdata)

            except Exception as e:
                print(f"Failed to add right texture {idx}: {e}")



    # Clear previous content inside main_window or create window if not exists
    if dpg.does_item_exist("main_window"):
        dpg.delete_item("main_window", children_only=True)
    else:
        dpg.add_window(tag="main_window", label="Image Comparison", width=window_width, height=window_height)

    with dpg.group(parent="main_window"):
        dpg.add_text(f"Comparing: {os.path.basename(left_folder)} vs {os.path.basename(right_folder)}", tag="status_text")
        button_width = (window_width - 40) // 2

        with dpg.group(horizontal=True):
            dpg.add_button(label="Left is Better", width=button_width, height=60,
                           callback=mark_left_as_better,
                           user_data={"left": left_folder, "right": right_folder})
            dpg.add_spacer(width=20)
            dpg.add_button(label="Right is Better", width=button_width, height=60,
                           callback=mark_right_as_better,
                           user_data={"left": left_folder, "right": right_folder})

        dpg.add_spacing(count=2)

        with dpg.child_window(width=-1, height=window_height - 140, autosize_x=True, horizontal_scrollbar=False):
            for idx in range(min_length):
                left_id = f"left_folder_{idx}"
                right_id = f"right_folder_{idx}"

                with dpg.group(horizontal=True):
                    dpg.add_image(left_id)
                    dpg.add_spacer(width=20)
                    dpg.add_image(right_id)
                dpg.add_spacer(height=15)

def next_round():
    global current_index, current_winner

    if current_index >= len(folders):
        if dpg.does_item_exist("main_window"):
            dpg.delete_item("main_window", children_only=True)
        dpg.add_window(tag="main_window", label="Result", width=600, height=200)
        dpg.add_text(f"✅ DONE — Winner: {current_winner}", parent="main_window")
        write_result(current_winner)
        return

    left = current_winner
    right = folders[current_index]
    current_index += 1
    create_gui(left, right)




folders = [line.strip() for line in sys.stdin if line.strip()]
# folders = ["_large_tests/8_0.95_3_5_48", "_large_tests/8_0.95_3_5_fin"]
current_winner = folders[0]

dpg.create_context()
dpg.create_viewport(title="Tournament Image Viewer", width=window_width, height=window_height)
dpg.setup_dearpygui()
dpg.show_viewport()
next_round()
dpg.start_dearpygui()
dpg.destroy_context()