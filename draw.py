import os
import sys
import argparse
from glob import glob

import cv2
import numpy as np
import pandas as pd

COLOR_GT = (0, 0, 255)  # Red
COLOR_PR = (0, 255, 0)  # Green


def get_all_unique_filenames(csv_files):
    all_filenames = set()
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"Info: '{csv_file}' not found, skipping.")
            continue
        
        data = pd.read_csv(csv_file)
        if data.empty:
            print(f"Info: '{csv_file}' is empty.")
            continue
        
        if "file_name" in data.columns:
            all_filenames.update(data["file_name"].unique())
        else:
            print(f"Warning: '{csv_file}' does not have a 'file_name' column.")
            
    return all_filenames


def make_blank_images_from_list(filenames_set, images_folder, img_size=(320, 320), ext=".png"):
    """
    Creates black background images based on the provided list of filenames (set).
    """
    print(f"Creating black background images in '{images_folder}' folder...")
    os.makedirs(images_folder, exist_ok=True)
    
    img_h, img_w = img_size
    count = 0
    if not filenames_set:
        print("Warning: No images to create.")
        return

    for file_name in filenames_set:
        base_name = os.path.splitext(str(file_name))[0]
        img_path = os.path.join(images_folder, f"{base_name}{ext}")

        if not os.path.exists(img_path):
            black = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            cv2.imwrite(img_path, black)
            count += 1
            
    print(f"Created {count} new background images.")


def draw_boxes(data_file, images_folder, color, ext=".png"):
    """
    Reads coordinates from a CSV file and draws bounding boxes on images.
    """
    print(f"Drawing boxes from '{data_file}' (Color: {color})...")
        
    data = pd.read_csv(data_file)

    drawn_count = 0
    for file_name in data["file_name"].unique():
        
        base_name = os.path.splitext(str(file_name))[0]
        img_path = os.path.join(images_folder, f"{base_name}{ext}")
        
        img = cv2.imread(img_path)

        rects = data[data["file_name"] == file_name]

        for _, row in rects.iterrows():
            center = (int(row["cx"]), int(row["cy"]))
            size = (int(row["width"]), int(row["height"]))
            
            rect = (center, size, 0) 
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(img, [box], 0, color, 2)

        cv2.imwrite(img_path, img)
        drawn_count += 1

    print(f"Finished processing '{data_file}'. Drew boxes on {drawn_count} images.")


def clean_directory(images_folder):
    """
    Deletes all files in the existing image folder.
    """
    print(f"Deleting existing result files in '{images_folder}' folder...")
    if os.path.exists(images_folder):
        files = glob(os.path.join(images_folder, "*"))
        count = 0
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
                count += 1
        print(f"Deleted {count} files.")
    else:
        print(f"'{images_folder}' folder does not exist. It will be created.")


def main(args):
    img_extension = ".png"

    gt_exists = os.path.exists(args.gt_file)
    pr_exists = os.path.exists(args.pr_file)
    
    # 1. Clean the existing results folder
    clean_directory(args.output_dir)

    # 2. Collect all unique filenames from both GT and PR files
    print("--- 1. Collecting image list ---")
    files_to_check = []
    if gt_exists:
        files_to_check.append(args.gt_file)
    if pr_exists:
        files_to_check.append(args.pr_file)
        
    all_filenames = get_all_unique_filenames(files_to_check)
    
    print(f"Processing a total of {len(all_filenames)} unique images: {all_filenames}")

    # 3. Create blank images for all unique filenames
    print("\n--- 2. Creating background images ---")
    make_blank_images_from_list(all_filenames, args.output_dir, ext=img_extension)
    
    # 4. Draw GT (Ground Truth) boxes (Red)
    print("\n--- 3. Drawing Ground Truth (GT) boxes ---")
    draw_boxes(args.gt_file, args.output_dir, COLOR_GT, ext=img_extension)

    # 5. Draw PR (Prediction) boxes (Green)
    print("\n--- 4. Drawing Prediction (PR) boxes ---")
    draw_boxes(args.pr_file, args.output_dir, COLOR_PR, ext=img_extension)
    
    print("\n--- Visualization complete ---")
    print(f"Results saved in '{args.output_dir}' folder.")
    print(f"Red: Ground Truth (GT) {COLOR_GT}")
    print(f"Green: Prediction (PR) {COLOR_PR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Ground Truth (GT) and Prediction (PR) bounding boxes.")
    
    parser.add_argument(
        "--gt_file", 
        type=str, 
        default="gt.csv", 
        help="Path to the Ground Truth (GT) CSV file"
    )
    parser.add_argument(
        "--pr_file", 
        type=str, 
        default="pr.csv", 
        help="Path to the Prediction (PR) CSV file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="images_results",
        help="Folder to save the resulting images"
    )
    
    args = parser.parse_args()
    main(args)
