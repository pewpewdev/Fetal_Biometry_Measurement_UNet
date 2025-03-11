# import pandas as pd
# from PIL import Image, ImageEnhance, ImageOps, ImageFile, ImageFilter
# import numpy as np
# import random
# import threading, os, time
# import logging

# logger = logging.getLogger(__name__)
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# class DataAugmentation:
#     """
#     Contains specific data augmentation methods: rotation, translation, and intensity scaling/shifting.
#     """

#     def __init__(self):
#         pass

#     @staticmethod
#     def openImage(image):
#         try:
#             return Image.open(image, mode="r")
#         except Exception as e:
#             logger.error(f"Failed to open image {image}: {str(e)}")
#             return None

#     @staticmethod
#     def randomIntensity(image, max_scale=0.1, max_shift=0.1):
#         """
#         Randomly scale and shift the intensity of the image.
#         :param image: PIL image object.
#         :param max_scale: Maximum scaling factor (default is 10%).
#         :param max_shift: Maximum shift factor (default is 10%).
#         :return: Image with adjusted intensity.
#         """
#         img_array = np.asarray(image).astype(np.float32)
        
#         # Random scaling and shifting
#         scale = 1 + np.random.uniform(-max_scale, max_scale)
#         shift = np.random.uniform(-max_shift, max_shift) * 255
        
#         img_array = img_array * scale + shift
#         img_array = np.clip(img_array, 0, 255).astype(np.uint8)  # Clip values to valid range
#         return Image.fromarray(img_array)

#     @staticmethod
#     def gaussianBlur(image, radius=0.5):
#         return image.filter(ImageFilter.GaussianBlur(radius))

#     @staticmethod
#     def saveImage(image, path):
#         try:
#             image.save(path)
#             logger.info(f"Image saved successfully: {path}")
#         except Exception as e:
#             logger.error(f"Failed to save image {path}: {str(e)}")


# def makeDir(path):
#     try:
#         if not os.path.exists(path):
#             if not os.path.isfile(path):
#                 os.makedirs(path)
#             return 0
#         else:
#             return 1
#     except Exception as e:
#         logger.error(f"Failed to create directory {path}: {str(e)}")
#         return -2


# def is_image_corrupted(image_path):
#     try:
#         with Image.open(image_path) as img:
#             img.verify()  # Verify that the file is not corrupted
#         return False
#     except Exception as e:
#         logger.error(f"Corrupted image: {image_path} - {str(e)}")
#         return True


# def imageOps(func_name, image, des_path, file_name, times=5):
#     funcMap = {
#         "randomIntensity": DataAugmentation.randomIntensity,
#         "gaussianBlur": DataAugmentation.gaussianBlur
#     }
#     if funcMap.get(func_name) is None:
#         logger.error("%s is not exist", func_name)
#         return -1

#     for _i in range(0, times, 1):
#         try:
#             new_image = funcMap[func_name](image)
#             output_path = os.path.join(des_path, file_name)
#             logger.info(f"Saving image to: {output_path}")
#             DataAugmentation.saveImage(new_image, output_path)
#         except Exception as e:
#             logger.error(f"Failed to process image {file_name} with {func_name}: {str(e)}")


# opsList = {"randomIntensity","gaussianBlur"}  


# def threadOPS(path, new_path, ground_truth_csv, new_ground_truth_csv, max_augmented_images=400):
#     """
#     Multi-threaded processing of images.
#     :param path: Source directory.
#     :param new_path: Destination directory.
#     :param ground_truth_csv: Path to the original ground truth CSV file.
#     :param new_ground_truth_csv: Path to save the new ground truth CSV file.
#     :param max_augmented_images: Maximum number of augmented images to generate.
#     :return:
#     """
#     if os.path.isdir(path):
#         img_names = os.listdir(path)
#     else:
#         img_names = [path]

#     # Load the original ground truth CSV
#     ground_truth_df = pd.read_csv(ground_truth_csv)
#     new_ground_truth_data = []

#     augmented_image_count = 0

#     for img_name in img_names:
#         logger.info(f"Processing image: {img_name}")
#         tmp_img_name = os.path.join(path, img_name)
#         if not os.path.exists(tmp_img_name):
#             logger.error(f"File not found: {tmp_img_name}")
#             continue
#         if is_image_corrupted(tmp_img_name):
#             logger.error(f"Corrupted image: {tmp_img_name}")
#             continue

#         if os.path.isdir(tmp_img_name):
#             if makeDir(os.path.join(new_path, img_name)) != -1:
#                 threadOPS(tmp_img_name, os.path.join(new_path, img_name), ground_truth_csv, new_ground_truth_csv, max_augmented_images)
#             else:
#                 logger.error('Failed to create new directory')
#                 return -1
#         elif tmp_img_name.split('.')[1] != "DS_Store":
#             # Read and process the image
#             image = DataAugmentation.openImage(tmp_img_name)
#             if image is None:
#                 continue

#             threadImage = [0] * 5
#             _index = 0
#             for ops_name in opsList:
#                 threadImage[_index] = threading.Thread(target=imageOps,
#                                                        args=(ops_name, image, new_path, img_name,))
#                 threadImage[_index].start()
#                 _index += 1
#                 time.sleep(0.2)

#                 # Get the ground truth for the original image
#                 original_image_name = img_name
#                 ground_truth = ground_truth_df[ground_truth_df['image_name'] == original_image_name]

#                 if not ground_truth.empty:
#                     for _i in range(0, 5, 1):
#                         augmented_image_name = img_name
#                         # Copy the entire row of the original ground truth
#                         new_ground_truth_row = ground_truth.iloc[0].copy()
#                         # Update the image name to the augmented image name
#                         new_ground_truth_row['image_name'] = augmented_image_name
#                         # Append the new row to the ground truth data
#                         new_ground_truth_data.append(new_ground_truth_row)
#                         augmented_image_count += 1

#                         if augmented_image_count >= max_augmented_images:
#                             break

#                 if augmented_image_count >= max_augmented_images:
#                     break

#             if augmented_image_count >= max_augmented_images:
#                 break

#     # Save the new ground truth CSV
#     new_ground_truth_df = pd.DataFrame(new_ground_truth_data)
#     new_ground_truth_df.to_csv(new_ground_truth_csv, index=False)


# # Define input and output directories
# input_dir = "selected_image_clahe"
# output_dir = "/home/prerna/landmark_detection/selected_image_aug"

# # Define ground truth CSV paths
# ground_truth_csv = "selected_image_ground.csv"
# new_ground_truth_csv = "selected_image_aug_ground.csv"

# # Create output directory if it doesn't exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Start the augmentation process
# threadOPS(input_dir, output_dir, ground_truth_csv, new_ground_truth_csv, max_augmented_images=400)


import pandas as pd
from PIL import Image, ImageEnhance, ImageOps, ImageFile, ImageFilter
import numpy as np
import random
import threading, os, time
import logging

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DataAugmentation:
    @staticmethod
    def openImage(image_path):
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image {image_path}: {str(e)}")
            return None

    @staticmethod
    def randomIntensity(image, max_scale=0.1, max_shift=0.1):
        img_array = np.asarray(image).astype(np.float32)
        scale = 1 + np.random.uniform(-max_scale, max_scale)
        shift = np.random.uniform(-max_shift, max_shift) * 255
        img_array = np.clip(img_array * scale + shift, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    @staticmethod
    def gaussianBlur(image, radius=0.5):
        return image.filter(ImageFilter.GaussianBlur(radius))

    @staticmethod
    def saveImage(image, path):
        try:
            image.save(path)
            logger.info(f"Image saved successfully: {path}")
        except Exception as e:
            logger.error(f"Failed to save image {path}: {str(e)}")

def is_image_corrupted(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return False
    except Exception:
        return True

def imageOps(func_name, image, des_path, file_name):
    funcMap = {
        "randomIntensity": DataAugmentation.randomIntensity,
        "gaussianBlur": DataAugmentation.gaussianBlur
    }

    if func_name not in funcMap:
        logger.error("%s is not a valid function", func_name)
        return

    try:
        new_image = funcMap[func_name](image)
        output_path = os.path.join(des_path, f"{func_name}_{file_name}")
        DataAugmentation.saveImage(new_image, output_path)
    except Exception as e:
        logger.error(f"Error processing image {file_name} with {func_name}: {str(e)}")

def threadOPS(path, new_path, ground_truth_csv, new_ground_truth_csv, max_augmented_images=400):
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    img_names = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not img_names:
        logger.error(f"No valid images found in {path}")
        return

    ground_truth_df = pd.read_csv(ground_truth_csv)
    new_ground_truth_data = []
    augmented_image_count = 0

    for img_name in img_names:
        img_path = os.path.join(path, img_name)

        if is_image_corrupted(img_path):
            logger.warning(f"Skipping corrupted image: {img_path}")
            continue

        image = DataAugmentation.openImage(img_path)
        if image is None:
            continue

        threads = []
        for ops_name in {"randomIntensity", "gaussianBlur"}:
            t = threading.Thread(target=imageOps, args=(ops_name, image, new_path, img_name,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()  # Ensure all threads complete before moving to the next image

        original_gt = ground_truth_df[ground_truth_df['image_name'] == img_name]
        if not original_gt.empty:
            for ops_name in {"randomIntensity", "gaussianBlur"}:
                augmented_image_name = f"{ops_name}_{img_name}"
                new_gt_row = original_gt.iloc[0].copy()
                new_gt_row['image_name'] = augmented_image_name
                new_ground_truth_data.append(new_gt_row)
                augmented_image_count += 1

        if augmented_image_count >= max_augmented_images:
            break

    pd.DataFrame(new_ground_truth_data).to_csv(new_ground_truth_csv, index=False)
    logger.info(f"Saved new ground truth CSV: {new_ground_truth_csv}")

# Paths
input_dir = "selected_image_clahe"
output_dir = "/home/prerna/landmark_detection/selected_image_aug"
ground_truth_csv = "selected_image_ground.csv"
new_ground_truth_csv = "selected_image_aug_ground.csv"

threadOPS(input_dir, output_dir, ground_truth_csv, new_ground_truth_csv, max_augmented_images=400)
