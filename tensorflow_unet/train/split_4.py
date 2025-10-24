import os
import random

# === Configuration ===
source_dir = 'dataset/raw_images/a'
source_dir = 'dataset/raw_images/b'
source_dir = 'dataset/raw_images/merge'
source_dir = 'dataset/raw_images/new_merge'
source_dir = 'dataset/raw_images/test'
source_dir = 'dataset/raw_images/valid'

mask_dir = source_dir + '_mask'
output_base_dir = 'dataset/split_images'

# Split proportions (out of 10)
splits = {
    'train': 3,
    'val': 2,
    'test': 1
}

print(f"🔍 Source image dir: {source_dir}")
print(f"🔍 Mask dir: {mask_dir}")
print(f"📁 Output base dir: {output_base_dir}")

# === Create output directories ===
for split in splits:
    img_dir = os.path.join(output_base_dir, split)
    mask_out_dir = os.path.join(output_base_dir, split + '_mask')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)
    print(f"📁 Created: {img_dir}, {mask_out_dir}")

# === Filter files that have masks ===
print("🔎 Scanning image directory...")
all_files = sorted([
    f for f in os.listdir(source_dir)
    if os.path.isfile(os.path.join(source_dir, f))
])

valid_files = []
for f in all_files:
    name, ext = os.path.splitext(f)
    mask_filename = f"{name}_pixels0.png"
    mask_path = os.path.join(mask_dir, mask_filename)
    if os.path.exists(mask_path):
        valid_files.append(f)
        print(f"✅ Found mask for image: {f} → {mask_filename}")
    else:
        print(f"⛔ No mask found for image: {f} → Skipping.")

# === Shuffle and split ===
random.shuffle(valid_files)
n = len(valid_files)
n_test = 0
n_val =  n // 15
n_train = n - n_test - n_val

# n_test = n // 10
# n_val =  n // 10
# n_train = n - n_test - n_val


test_files = valid_files[:n_test]
val_files = valid_files[n_test:n_test + n_val]
train_files = valid_files[n_test + n_val:n_test + n_val + n_train]

# === Link creation ===
def link_with_mask(file_list, split_name):
    print(f"\n🔗 Creating links for {split_name}...")
    for f in file_list:
        name, ext = os.path.splitext(f)
        mask_filename = f"{name}_pixels0.png"

        image_src = os.path.abspath(os.path.join(source_dir, f))
        image_dst = os.path.abspath(os.path.join(output_base_dir, split_name, f))
        mask_src = os.path.abspath(os.path.join(mask_dir, mask_filename))
        mask_dst = os.path.abspath(os.path.join(output_base_dir, split_name + '_mask', mask_filename))



        # image_src = os.path.relpath(os.path.join(source_dir, f), os.path.join(output_base_dir, split_name))
        # image_dst = os.path.join(output_base_dir, split_name, f)

        # mask_src = os.path.relpath(os.path.join(mask_dir, mask_filename), os.path.join(output_base_dir, split_name + '_mask'))
        # mask_dst = os.path.join(output_base_dir, split_name + '_mask', mask_filename)

        try:
            
            
            os.symlink(image_src, image_dst)
            os.symlink(mask_src, mask_dst)
            print(f"[+] Linked: {f} and {mask_filename}")
        except FileExistsError:
            print(f"[-] Skipped (already linked): {f}")
        except Exception as e:
            print(f"[!] Error linking {f}: {e}")

# Execute linking
link_with_mask(train_files, 'train')
link_with_mask(val_files, 'val')
link_with_mask(test_files, 'test')


print(f"\n[+] Total valid images: {n}")
print(f"[+] Train: {n_train}, Val: {n_val}, Test: {n_test}, Sum: {n_test + n_val + n_train}")


print("\n🎉 All done. Only images with masks were processed.")
