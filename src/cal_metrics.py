import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import subprocess
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示 FATAL 错误，彻底屏蔽

import re
import numpy as np
import pandas as pd
import imageio.v3 as iio
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.util import img_as_float32
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal import CLIPImageQualityAssessment

# import tensorflow as tf
# from DBI.predict import compute_dbi_image

# === DEVICE ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === get dir ===
def resolve_debanded_dir(deband_name, dataset_name):
    if deband_name == "Banded_Input":
        return {
            "deepDeband": "/media/on23019/server1/video_dataset/debanding_dataset/deepDeband/split/test/banded",
            # "BAND-2k": "/media/on23019/server1/video_dataset/debanding_dataset/BAND-2k/Image_source/quantified_png"
            "BAND-2k": "/media/xinyi/server/video_dataset/debanding_dataset/BAND-2k/Image_source/quantified_png",
            "HD_images_dataset_dbi": "/media/xinyi/server/video_dataset/debanding_dataset/HD_images_dataset_dbi/Quantized_Images"
        }.get(dataset_name)

    elif deband_name in {
        "WaveMamba", "WaveMamba_wavelet", "WaveMamba_nowarmup", "WaveMamba_nowarmup_wavelet", "WaveMamba_less_iter", "WaveMamba_less_iter_wavelet"
    }:
        return f"/home/xinyi/Project/Debanding/Wave-Mamba/results/{dataset_name}/{deband_name}"

    elif deband_name == "deepDeband-f":
        return "/home/xinyi/Project/Debanding/WaveMamba-dwt/test_results/HD_images_dataset_dbi/deepDeband-f"

    elif deband_name == "other":
        return "/home/xinyi/Project/Debanding/test_img/debanded"

    elif deband_name == "eval":
        deband_name = 'debanded_with_ldm'
        return f"/media/on23019/server1/video_dataset/debanding_dataset/debanding_evaluation/{deband_name}"

    else:
        return f"/home/xinyi/Project/Debanding/WaveMamba-dwt/test_results/{dataset_name}/{deband_name}"

def resolve_pristine_dir(dataset_name):
    return {
        "deepDeband": "/media/on23019/server1/video_dataset/debanding_dataset/deepDeband/split/test/pristine",
        # "BAND-2k": "/media/on23019/server1/video_dataset/debanding_dataset/BAND-2k/Image_source/Origin_image",
        "BAND-2k": "/media/xinyi/server/video_dataset/debanding_dataset/BAND-2k/Image_source/Origin_image",
        "HD_images_dataset_dbi": "/media/xinyi/server/video_dataset/debanding_dataset/HD_images_dataset_dbi/Original_Images",
        "other": "/home/xinyi/Project/Debanding/test_img/pristine",
        "eval": "/media/on23019/server1/video_dataset/debanding_dataset/deepDeband/split/test/pristine"
    }.get(dataset_name)

def resolve_bband_csv(deband_name, dataset_name):
    if dataset_name == "eval":
        deband_name = 'debanded_with_ldm'
        return f"./BBAND/deepDeband_eval/banding_scores_{deband_name}.csv"
    elif dataset_name == "other":
        return f"./BBAND/deepDeband/banding_scores_{deband_name}.csv"
    else:
        return f"./BBAND/{dataset_name}/banding_scores_{deband_name}.csv"

# metric wrappers
def compute_psnr(img1, img2):
    return psnr(img1, img2, data_range=1.0)

def compute_ssim(img1, img2):
    if img1.ndim == 3 and img1.shape[2] == 3: # RGB image
        img1 = np.mean(img1, axis=2) # Convert to grayscale
        img2 = np.mean(img2, axis=2)
    return ssim(img1, img2, data_range=1.0)

def compute_lpips(pris_path, deb_path):
    pristine_img = Image.open(pris_path).convert('RGB')
    debanded_img = Image.open(deb_path).convert('RGB')

    # convert images to numpy arrays
    pristine_img_array = np.array(pristine_img).astype(np.float32)
    debanded_img_array = np.array(debanded_img).astype(np.float32)

    # normalize images to [-1, 1]
    pristine_img_array = (pristine_img_array / 127.5) - 1.0
    debanded_img_array = (debanded_img_array / 127.5) - 1.0
    # convert images to tensors and change channel order (C, H, W)
    pristine_img_tensor = torch.tensor(pristine_img_array).permute(2, 0, 1).unsqueeze(0)
    debanded_img_tensor = torch.tensor(debanded_img_array).permute(2, 0, 1).unsqueeze(0)
    pristine_tensor = pristine_img_tensor.to(DEVICE)
    debanded_tensor = debanded_img_tensor.to(DEVICE)
    with torch.no_grad():
        return lpips_model(pristine_tensor, debanded_tensor).item()

def compute_cambi_image(pris_path, deb_path, tmp_dir="./temp_yuvs", vmaf_path="/home/xinyi/Project/Debanding/vmaf/libvmaf/build/tools/vmaf"):
    """Compute CAMBI using VMAF CLI from pristine and debanded image paths."""
    os.makedirs(tmp_dir, exist_ok=True)
    yuv_pris_path = os.path.join(tmp_dir, "pris.yuv")
    yuv_deb_path = os.path.join(tmp_dir, "deb.yuv")

    try:
        # Convert pristine image to YUV
        ffmpeg_cmd_pris = [
            "ffmpeg", "-y", "-i", pris_path, "-s", "256x256",
            "-pix_fmt", "yuv420p", "-frames:v", "1", yuv_pris_path
        ]
        subprocess.run(ffmpeg_cmd_pris, check=True, capture_output=True)
        # Convert debanded image to YUV
        ffmpeg_cmd_deb = [
            "ffmpeg", "-y", "-i", deb_path, "-s", "256x256",
            "-pix_fmt", "yuv420p", "-frames:v", "1", yuv_deb_path
        ]
        subprocess.run(ffmpeg_cmd_deb, check=True, capture_output=True)

        # Run VMAF to get CAMBI
        vmaf_cmd = [
            vmaf_path,
            "--reference", yuv_pris_path,
            "--distorted", yuv_deb_path,
            "--width", "256", "--height", "256",
            "--pixel_format", "420",
            "--bitdepth", "8",
            "--no_prediction",
            "--feature", "cambi",
            "--output", "/dev/stdout"
        ]
        result = subprocess.run(vmaf_cmd, capture_output=True, text=True, check=True)
        # Parse CAMBI score
        for line in result.stdout.splitlines():
            if '<metric name="cambi"' in line:
                # print(line)
                match = re.search(r'mean="([\d.]+)"', line)
                if match:
                    return float(match.group(1))
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
    finally:
        if os.path.exists(yuv_pris_path):
            os.remove(yuv_pris_path)
        if os.path.exists(yuv_deb_path):
            os.remove(yuv_deb_path)
    return None

if __name__ == "__main__":

    # deband_name = "Banded_Input"
    # WaveMamba + wavelet map (separate)
    # deband_name = "WaveMamba"
    deband_name = "WaveMamba_wavelet"
    # WaveMamba + wavelet map (combined)
    # deband_name = "WaveMamba-map"
    # deband_name = "WaveMamba-dwt"
    # deband_name = "deepDeband-f"
    # deband_name = "other"
    # deband_name = "eval"

    dataset_name = "HD_images_dataset_dbi"

    debanded_dir = resolve_debanded_dir(deband_name, dataset_name)
    pristine_dir = resolve_pristine_dir(dataset_name)
    bband_csv_path = resolve_bband_csv(deband_name, dataset_name)
    output_csv = f"./combined_metrics/combined_metrics_{deband_name}.csv"

    # get BBAND
    bband_df = pd.read_csv(bband_csv_path)
    bband_df.rename(columns={"ImageName": "filename", "BandingScore": "BBAND"}, inplace=True)
    bband_df["filename"] = bband_df["filename"].str.strip()
    bband_dict = dict(zip(bband_df["filename"], bband_df["BBAND"]))

    # load model
    lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(DEVICE)
    clip_iqa_metric = CLIPImageQualityAssessment(prompts=("quality", "noisiness")).to(DEVICE)
    # dbi_model = tf.keras.models.load_model('/home/xinyi/Project/Debanding/src/DBI/CNN_classifier/')

    # main
    results = []
    for filename in sorted(os.listdir(debanded_dir)):
        if filename.lower().endswith(".png"):
            deb_path = os.path.join(debanded_dir, filename)
            pris_path = os.path.join(pristine_dir, filename)
            print(deb_path)
            print(pris_path)

            if os.path.exists(pris_path):
                print(f"Processing {filename}")
                pristine = img_as_float32(iio.imread(pris_path))
                deband = img_as_float32(iio.imread(deb_path))

                psnr_val  = compute_psnr(pristine, deband)
                ssim_val  = compute_ssim(pristine, deband)
                lpips_val = compute_lpips(pris_path, deb_path)
                bband_val = bband_dict.get(filename, None)

                # CAMBI
                cambi_val = compute_cambi_image(pris_path, deb_path)
                # print(cambi_val)

                # DBI
                # dbi_val = compute_dbi_image(deb_path, dbi_model)
                # print(dbi_val)

                # CLIP-IQA
                clip_img = torch.tensor(deband * 255).clamp(0, 255).byte()
                if clip_img.ndim == 3 and clip_img.shape[2] == 3:
                    clip_img = clip_img.permute(2, 0, 1)
                else:
                    clip_img = clip_img.unsqueeze(0).repeat(3, 1, 1)
                clip_img = clip_img.to(DEVICE)
                with torch.no_grad():
                    clip_scores = clip_iqa_metric(clip_img.unsqueeze(0))

                results.append({
                    "filename": filename,
                    "PSNR": psnr_val,
                    "SSIM": ssim_val,
                    "LPIPS": lpips_val,
                    "CAMBI": cambi_val,
                    "BBAND": bband_val,
                    # "DBI": dbi_val,
                    "CLIP-IQA (quality)": clip_scores["quality"].item(),
                    "CLIP-IQA (noisiness)": clip_scores["noisiness"].item(),
                })

            else:
                print(f"Skipping {filename}: Processed image not found.")

    # Save combined CSV
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"All metrics saved to {output_csv}")