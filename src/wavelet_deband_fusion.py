import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
from skimage import io, color, img_as_float, img_as_ubyte
from skimage.transform import resize
import pywt
import matplotlib.pyplot as plt

def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

def compute_wavelet_detail(image_gray, wavelet='haar', levels=5, resize_to=(256, 256)):
    coeffs = pywt.wavedec2(image_gray, wavelet=wavelet, level=levels)
    cH, cV, cD = coeffs[levels]
    detail = np.abs(cH) + np.abs(cV) + np.abs(cD)
    detail = normalize(detail)
    detail_resized = resize(detail, resize_to, anti_aliasing=True)
    return detail_resized

def fuse_images(banded_image, debanded_image, wavelet_detail, threshold_ratio=0.2, use_threshold=True, use_half_weight=True):
    # Resize wavelet_detail to match image shape
    H, W, C = banded_image.shape
    wavelet_detail_resized = resize(wavelet_detail, (H, W), anti_aliasing=True)
    wavelet_detail_resized = wavelet_detail_resized[:, :, np.newaxis]  # (H, W, 1)

    wavelet_max = np.max(wavelet_detail_resized)
    if wavelet_max == 0:
        print("Wavelet detail is zero. Using debanded image as output.")
        return np.clip(debanded_image, 0, 255).astype(np.uint8)

    weight = wavelet_detail_resized / wavelet_max
    if use_half_weight:
        weight = weight / 2

    if use_threshold:
        threshold = threshold_ratio
        mask = (wavelet_detail_resized > threshold * wavelet_max).astype(np.float32)
        alpha = weight * mask
    else:
        alpha = weight

    # Expand alpha to 3 channels if image is RGB
    if C == 3:
        alpha = np.repeat(alpha, 3, axis=2)

    output = debanded_image * (1 - alpha) + banded_image * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def process_image(banded_path, debanded_path, output_path, wavelet_output_path):
    banded = img_as_float(io.imread(banded_path))
    debanded = img_as_float(io.imread(debanded_path))
    gray = color.rgb2gray(banded)

    wavelet_detail = compute_wavelet_detail(gray)
    wavelet_filename = os.path.basename(output_path)
    wavelet_save_path = os.path.join(wavelet_output_path, wavelet_filename)
    io.imsave(wavelet_save_path, img_as_ubyte(wavelet_detail))

    output = fuse_images(banded * 255, debanded * 255, wavelet_detail)
    io.imsave(output_path, output)
    print(f"Processed: {os.path.basename(output_path)}")

def process_dataset(banded_folder, debanded_folder, output_folder, wavelet_output_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(wavelet_output_folder, exist_ok=True)
    files = sorted([f for f in os.listdir(debanded_folder) if f.lower().endswith('.png')])
    for filename in files:
        banded_path = os.path.join(banded_folder, filename)
        debanded_path = os.path.join(debanded_folder, filename)
        output_path = os.path.join(output_folder, filename)
        if os.path.exists(banded_path) and os.path.exists(debanded_path):
            process_image(banded_path, debanded_path, output_path, wavelet_output_folder)
        else:
            print(f"Skipping {filename}: file missing.")

if __name__ == "__main__":
    # test images
    # banded_folder = "../test_img/banded"
    # debanded_folder = "../test_img/WaveMamba"
    # output_folder = "../test_img/deepDeband_wavelet"
    # wavelet_output_folder = "../test_img/wavelet"

    # banded_folder = "/media/on23019/server1/video_dataset/debanding_dataset/deepDeband/split/test/banded"
    # banded_folder = "/media/xinyi/server/video_dataset/debanding_dataset/BAND-2k/Image_source/Origin_image"
    banded_folder = "/media/xinyi/server/video_dataset/debanding_dataset/HD_images_dataset_dbi/Original_Images"

    # WaveMamba + wavelet map (separate)
    debanded_folder = "/home/xinyi/Project/Debanding/WaveMamba-Frequency-Masking/test_results/HD_images_dataset_dbi/WaveMamba"
    output_folder = "/home/xinyi/Project/Debanding/WaveMamba-Frequency-Masking/test_results/HD_images_dataset_dbi/WaveMamba_wavelet"
    wavelet_output_folder = "/home/xinyi/Project/Debanding/WaveMamba-Frequency-Masking/test_results/HD_images_dataset_dbi/wavelet"

    process_dataset(banded_folder, debanded_folder, output_folder, wavelet_output_folder)
    print("All images processed.")
