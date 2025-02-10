import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_niblack, threshold_sauvola
from collections import defaultdict

def align_images(ref_image, img_to_align):
    """Aligns img_to_align to ref_image using only translation (small shifts)."""
    orb = cv2.ORB_create(500)  # Feature detector

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(ref_image, None)
    kp2, des2 = orb.detectAndCompute(img_to_align, None)

    # Use Brute-Force matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Use top N matches for translation estimation
    num_good_matches = max(10, int(len(matches) * 0.1))  # Use 10% of best matches
    good_matches = matches[:num_good_matches]

    # Ensure we have enough points to estimate translation
    if len(good_matches) < 3:
        print("Not enough good matches found, returning original image")
        return img_to_align  # Return unaligned image

    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Compute translation (shift) between images
    translation = np.mean(pts1 - pts2, axis=0)  # Average displacement

    # Create translation matrix
    M = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])

    # Apply translation shift only (no warping)
    aligned_img = cv2.warpAffine(img_to_align, M, (ref_image.shape[1], ref_image.shape[0]))

    return aligned_img

def resize_plate(plate: np.ndarray):
    image = plate

    # Target dimensions
    new_width = 300
    new_height = 75

    # Get original dimensions
    height, width, _ = image.shape

    # Determine scaling factor to fit within the target size
    scale_w = new_width / width
    scale_h = new_height / height
    scale = min(scale_w, scale_h)  # Choose the smaller scale to fit within the box

    # Compute new resized dimensions while maintaining aspect ratio
    resized_width = int(width * scale)
    resized_height = int(height * scale)

    # Resize the image
    resized_image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)

    # Create a blank black image with target size
    final_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Calculate padding offsets to center the image
    x_offset = (new_width - resized_width) // 2
    y_offset = (new_height - resized_height) // 2
    
    # Place the resized image in the center
    final_image[y_offset:y_offset+resized_height, x_offset:x_offset+resized_width] = resized_image

    return final_image

def fuse_thresholded_images(thresholded_images):
    """Aligns and fuses the last 5 thresholded images to enhance clarity."""
    
    # Ensure we have at least 5 images
    num_images = len(thresholded_images)
    if num_images < 2:
        print("⚠️ Less than 5 images available, using all available images.")
        selected_images = thresholded_images  # Use all if fewer than 5
    else:
        selected_images = thresholded_images[-2:]  # Select the last 5 images

    ref_image = selected_images[0]  # Use the first of the last 5 as reference
    aligned_images = [ref_image]

    # Align all selected images to the reference
    for img in selected_images[1:]:
        aligned_img = align_images(ref_image, img)  # Use translation-based alignment
        aligned_images.append(aligned_img)

    # Stack and compute the median fusion
    stacked_images = np.stack(aligned_images, axis=0)
    fused_image = np.median(stacked_images, axis=0).astype(np.uint8)

    return fused_image

def sharpenLAP(img: np.ndarray):
    # Define the Laplacian filters
    Lap = np.array([[0, 1, 0], 
                    [1, -4, 1], 
                    [0, 1, 0]], dtype=np.float32)

    strong_Lap = np.array([[-1, -1, -1], 
                            [-1,  8, -1], 
                            [-1, -1, -1]], dtype=np.float32)

    # Apply convolution with Laplacian filter
    a1 = cv2.filter2D(img, -1, Lap)
    a2 = np.clip(a1, 0, 255).astype(np.uint8)  # Normalize intensity

    # Compute first sharpened image
    sharpened1 = cv2.absdiff(img, a2)

    # Apply strong Laplacian filter
    a3 = cv2.filter2D(img, -1, strong_Lap)
    a4 = np.clip(a3, 0, 255).astype(np.uint8)  # Normalize intensity

    # Compute second sharpened image
    sharpened2 = cv2.add(img, a4)  # a + a4 (sharpening)

    return sharpened2

def sharpenHBF(img: np.ndarray):
    # Define the High Boost Filters (HBF)
    HBF = np.array([[0, -1,  0], 
                    [-1,  5, -1], 
                    [0, -1,  0]], dtype=np.float32)  # Central value = 5

    SHBF = np.array([[-1, -1, -1], 
                    [-1,  9, -1], 
                    [-1, -1, -1]], dtype=np.float32)  # Central value = 9

    # Apply convolution with High Boost Filters
    a1 = cv2.filter2D(img, -1, HBF)
    a2 = np.clip(a1, 0, 255).astype(np.uint8)  # Normalize intensity

    a3 = cv2.filter2D(img, -1, SHBF)
    a4 = np.clip(a3, 0, 255).astype(np.uint8)  # Normalize intensity

    return a2



def multi_threshold_plate(plate):
    # Convert to grayscale if necessary
    if len(plate.shape) == 3:
        plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE to enhance contrast (avoid over-enhancement)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(plate)

    # Apply different thresholding methods
    _, otsu_thresh = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_thresh = cv2.adaptiveThreshold(enhanced_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 21, 5)

    # Apply Niblack and Sauvola thresholding (LESS AGGRESSIVE)
    niblack_thresh = (enhanced_image > threshold_niblack(enhanced_image, window_size=25, k=0.1)).astype(np.uint8) * 255
    sauvola_thresh = (enhanced_image > threshold_sauvola(enhanced_image, window_size=25)).astype(np.uint8) * 255

    # Ensure consistent polarity (WHITE text on BLACK background)
    otsu_thresh = cv2.bitwise_not(otsu_thresh)
    adaptive_thresh = cv2.bitwise_not(adaptive_thresh)
    niblack_thresh = cv2.bitwise_not(niblack_thresh)
    sauvola_thresh = cv2.bitwise_not(sauvola_thresh)

    # Merge thresholded images using bitwise OR (STRONGER CHARACTER PRESERVATION)
    merged_thresh = cv2.bitwise_or(otsu_thresh, adaptive_thresh)
    merged_thresh = cv2.bitwise_or(merged_thresh, niblack_thresh)
    merged_thresh = cv2.bitwise_or(merged_thresh, sauvola_thresh)

    # Apply morphological opening (LESS AGGRESSIVE)
    kernel = np.ones((2, 2), np.uint8)  # Changed from (3,3) → (2,2)
    cleaned_thresh = cv2.morphologyEx(merged_thresh, cv2.MORPH_OPEN, kernel)   
        
    edges = cv2.Canny(plate, 100, 200)  # You can adjust the thresholds (100, 200) for fine-tuning
    refined_image = cv2.bitwise_and(cleaned_thresh, cleaned_thresh, mask=cv2.bitwise_not(edges))

    return refined_image


def weighted_majority_vote(ocr_results, entering=True):
    N = len(ocr_results)
    if N == 0:
        return "UNKNOWN"

    char_votes = defaultdict(lambda: defaultdict(float))  # {position: {char: weighted_score}}

    # Compute exponential decay weights
    exp_weights = np.exp(np.arange(N) - (N - 1) if entering else (N - 1) - np.arange(N))
    weights = exp_weights / np.sum(exp_weights)  # Normalize weights to sum to 1

    # Determine max length to handle short OCR results
    max_length = max(len(plate) for plate in ocr_results)

    # Apply weighted voting
    for i, plate in enumerate(ocr_results):
        for k, char in enumerate(plate):
            char_votes[k][char] += weights[i]

    # Get the most probable character for each position
    final_plate = "".join(
        max(char_votes[k], key=char_votes[k].get, default="?") if k in char_votes else "?"
        for k in range(max_length)
    )

    return final_plate if final_plate.strip("?") else "UNKNOWN"