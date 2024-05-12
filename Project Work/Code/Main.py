import os
import cv2
import numpy as np
from skimage import io, morphology, measure
from skimage.filters import threshold_local
from skimage.feature import canny


# Function to preprocess and segment images using adaptive thresholding with edge detection


def segment_image(image):
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (255 * image).astype(np.uint8)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_image = clahe.apply(image)

    # Apply edge detection using Canny edge detector
    edges = cv2.Canny(contrast_enhanced_image, 50, 150)

    # Apply adaptive thresholding
    block_size = 25  # Define block size for local thresholding
    adaptive_thresh = threshold_local(
        contrast_enhanced_image, block_size, offset=10)
    binary_image = contrast_enhanced_image > adaptive_thresh

    # Combine edge-detected image with thresholded image
    segmented_image = np.logical_or(edges, binary_image)

    # Morphological opening to remove small objects
    selem = morphology.disk(2)  # Disk size affects the size of objects removed
    opened_image = morphology.binary_opening(segmented_image, selem)

    # Label the objects
    labeled_image = measure.label(opened_image)

    # Remove small objects (adjust 'min_size' as needed)
    cleaned_image = morphology.remove_small_objects(
        labeled_image, min_size=150)

    # To ensure the output is binary (True/False), we convert back to bool type
    cleaned_binary_image = cleaned_image > 0

    return cleaned_binary_image

# Function to load datasets

def load_dataset(raw_images_dir):
    raw_images = []
    for img_name in sorted(os.listdir(raw_images_dir)):
        img_path = os.path.join(raw_images_dir, img_name)
        img = io.imread(img_path, as_gray=True)
        raw_images.append(img)
    return raw_images


raw_images_path = "BU-BIL_Dataset2/RawImages/chian1" # Location of the dataset
gold_standard_path = "BU-BIL_Dataset2/GoldStandard/chian1" #Location of the Ground-truth Files
images = load_dataset(raw_images_path)

# Segment images
segmented_images = [segment_image(image) for image in images]

for idx, segmented_image in enumerate(segmented_images):
    output_path = os.path.join("dataset2/chian1", f"segmented_{idx}.png")
    cv2.imwrite(output_path, (segmented_image.astype(np.uint8) * 255))


def calculate_iou(ground_truth, prediction):
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# Function to calculate Dice Coefficient


def calculate_dice(ground_truth, prediction):
    intersection = np.logical_and(ground_truth, prediction)
    dice_score = 2 * np.sum(intersection) / \
        (np.sum(ground_truth) + np.sum(prediction))
    return dice_score

# Function to evaluate segmentation against ground truth


def evaluate_segmentation(segmented_images, gold_standard_path):
    iou_scores = []
    dice_scores = []

    # Iterate through segmented images
    for idx, segmented_image in enumerate(segmented_images):
       
        gold_std_filename = f"{idx}.png"
        gold_std_path = os.path.join(gold_standard_path, gold_std_filename)

        # Check if the ground truth file exists
        if os.path.exists(gold_std_path):
            ground_truth = io.imread(
                gold_std_path, as_gray=True) > 0  # Binarize

            # Calculate IoU and Dice scores
            iou_score = calculate_iou(ground_truth, segmented_image)
            dice_score = calculate_dice(ground_truth, segmented_image)

            # Append the scores to the list
            iou_scores.append(iou_score)
            dice_scores.append(dice_score)

            print(
                f"Image {idx}: IoU = {iou_score:.4f}, Dice = {dice_score:.4f}")
        else:
            print(f"Ground truth for image {idx} not found.")

    # Calculate average scores
    avg_iou = np.mean(iou_scores) if iou_scores else 0
    avg_dice = np.mean(dice_scores) if dice_scores else 0
    print(f"Average IoU across all images: {avg_iou:.4f}")
    print(f"Average Dice across all images: {avg_dice:.4f}")

evaluate_segmentation(segmented_images, gold_standard_path)
