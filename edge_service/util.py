import cv2
import numpy as np

def align_images(ref_image, img_to_align):
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(ref_image, None)
    kp2, des2 = orb.detectAndCompute(img_to_align, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    num_good_matches = max(10, int(len(matches) * 0.1))
    good_matches = matches[:num_good_matches]
    if len(good_matches) < 3:
        return img_to_align
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    translation = np.mean(pts1 - pts2, axis=0)
    M = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
    aligned_img = cv2.warpAffine(img_to_align, M, (ref_image.shape[1], ref_image.shape[0]))
    return aligned_img

def resize_plate(plate: np.ndarray):
    new_width, new_height = 300, 75
    height, width, _ = plate.shape
    scale = min(new_width / width, new_height / height)
    resized_width, resized_height = int(width * scale), int(height * scale)
    resized_image = cv2.resize(plate, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)
    final_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    x_offset, y_offset = (new_width - resized_width) // 2, (new_height - resized_height) // 2
    final_image[y_offset:y_offset+resized_height, x_offset:x_offset+resized_width] = resized_image
    return final_image

def sharpenLAP(img: np.ndarray):
    Lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    a1 = cv2.filter2D(img, -1, Lap)
    a2 = np.clip(a1, 0, 255).astype(np.uint8)
    sharpened1 = cv2.absdiff(img, a2)
    strong_Lap = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    a3 = cv2.filter2D(img, -1, strong_Lap)
    a4 = np.clip(a3, 0, 255).astype(np.uint8)
    sharpened2 = cv2.add(img, a4)
    return sharpened2

def sharpenHBF(img: np.ndarray):
    HBF = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    a1 = cv2.filter2D(img, -1, HBF)
    return np.clip(a1, 0, 255).astype(np.uint8)
