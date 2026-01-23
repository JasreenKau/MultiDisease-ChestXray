import os
import cv2
from tqdm import tqdm

input_dir = "data/images"
output_dir = "data/images_resized"
target_size = (224, 224)

os.makedirs(output_dir, exist_ok=True)

image_files = [f for f in os.listdir(input_dir) if f.endswith(".png")]

print(f"Resizing {len(image_files)} images...")

for img_name in tqdm(image_files):
    input_path = os.path.join(input_dir, img_name)
    output_path = os.path.join(output_dir, img_name)

    try:
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, resized, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    except Exception as e:
        print(f"Error processing {img_name}: {e}")

print("✅ All images resized and compressed.")