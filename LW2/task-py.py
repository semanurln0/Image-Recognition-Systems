

import os
import shutil
import cv2
import numpy as np

RESULTS_DIR = 'results'
IMG_SIZE = 128
AUGS_PER_IMAGE = 5


def clear_results():
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR)



def get_dog_images(folder):
    images = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('jpg', 'jpeg', 'png')):
                images.append(os.path.join(root, f))
    return images

def resize(img):
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def to_lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

def random_rotate(img):
    angle = np.random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), angle, 1)
    return cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))

def random_flip(img):
    flip_code = np.random.choice([-1, 0, 1])
    return cv2.flip(img, flip_code)

def random_brightness(img):
    factor = np.random.uniform(0.7, 1.3)
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

def median_blur(img):
    k = np.random.choice([3, 5])
    return cv2.medianBlur(img, k)

def bilateral_filter(img):
    k = np.random.choice([3, 5])
    return cv2.bilateralFilter(img, k, 75, 75)

def gaussian_noise(img):
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def augment(img):
    img = random_rotate(img)
    img = random_flip(img)
    img = random_brightness(img)
    img = median_blur(img)
    img = bilateral_filter(img)
    img = gaussian_noise(img)
    return img

def save_augmented(img, base, idx):
    cv2.imwrite(f'{RESULTS_DIR}/{base}_aug{idx}.png', img)



def process_dataset(n_aug=AUGS_PER_IMAGE):
    img_paths = get_dog_images('dataset')
    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        img = resize(img)
        gray = to_grayscale(img)
        hsv = to_hsv(img)
        lab = to_lab(img)
        base = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(f'{RESULTS_DIR}/{base}_gray.png', gray)
        cv2.imwrite(f'{RESULTS_DIR}/{base}_hsv.png', cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        cv2.imwrite(f'{RESULTS_DIR}/{base}_lab.png', cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
        for i in range(n_aug):
            aug = augment(img.copy())
            save_augmented(aug, base, i)

if __name__ == '__main__':
    clear_results()
    if not os.path.exists('dataset'):
        raise FileNotFoundError("'dataset' folder not found. Please download and extract the dataset manually.")
    process_dataset()
