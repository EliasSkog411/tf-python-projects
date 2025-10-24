import os
from PIL import Image
import numpy as np
import dearpygui.dearpygui as dpg

SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

def get_image_files(folder_path):
    return sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(SUPPORTED_FORMATS)
    ])

def load_image(path, max_size=(300, 300)):
    """Load image from disk and prepare it for DPG texture registry."""
    try:
        img = Image.open(path).convert("RGBA")
        img.thumbnail(max_size)
        width, height = img.size
        image_data = np.array(img) / 255.0
        flat_data = image_data.flatten().astype(np.float32)
        return width, height, flat_data
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None, None, None

def create_gui(left_folder, right_folder):
    left_images = get_image_files(left_folder)
    right_images = get_image_files(right_folder)

    if len(left_images) != len(right_images):
        print("Error: Folder lengths do not match.")
        return

    dpg.create_context()
    dpg.create_viewport(title="Side-by-Side Image Viewer", width=1000, height=800)

    with dpg.texture_registry(show=False):
        for idx, (l_path, r_path) in enumerate(zip(left_images, right_images)):
            lw, lh, ldata = load_image(l_path)
            rw, rh, rdata = load_image(r_path)

            if lw and rw:
                dpg.add_static_texture(lw, lh, ldata, tag=f"left_texture_{idx}")
                dpg.add_static_texture(rw, rh, rdata, tag=f"right_texture_{idx}")

    with dpg.window(label="Image Comparison", width=1000, height=800):
        dpg.add_text("Scroll down to compare image pairs")
        with dpg.child_window(width=-1, height=700, autosize_x=True, horizontal_scrollbar=False):
            for idx in range(len(left_images)):
                with dpg.group(horizontal=True):
                    dpg.add_image(f"left_texture_{idx}")
                    dpg.add_spacer(width=20)
                    dpg.add_image(f"right_texture_{idx}")
                dpg.add_spacer(height=10)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

# --- Entry point ---
if __name__ == "__main__":
    # Replace these with your actual folders
    left_folder = "test"
    right_folder = "test"

    if not os.path.isdir(left_folder) or not os.path.isdir(right_folder):
        print("Error: Invalid folder paths")
    else:
        create_gui(left_folder, right_folder)
