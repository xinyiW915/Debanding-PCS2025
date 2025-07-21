import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

def generate_comparison_figure(debanded_path, fused_path, save_path):
    debanded = io.imread(debanded_path).astype(np.float32)
    fused = io.imread(fused_path).astype(np.float32)

    diff = np.abs(fused - debanded).mean(axis=2)
    diff_normalized = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

    mse = mean_squared_error(debanded, fused)
    psnr = peak_signal_noise_ratio(debanded, fused, data_range=255)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    axes[0].imshow(debanded.astype(np.uint8))
    axes[0].set_title("Debanded")
    axes[0].axis('off')

    axes[1].imshow(fused.astype(np.uint8))
    axes[1].set_title("Wavelet-Debanded")
    axes[1].axis('off')

    im = axes[2].imshow(diff_normalized, cmap='viridis')
    axes[2].set_title("Difference Heatmap")
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(f"MSE = {mse:.2f}, PSNR = {psnr:.2f} dB", fontsize=12)
    plt.savefig(save_path, dpi=150)
    plt.close()

def process_comparison_batch(debanded_folder, fused_folder, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    files = sorted([f for f in os.listdir(debanded_folder) if f.lower().endswith('.png')])
    for filename in files:
        debanded_path = os.path.join(debanded_folder, filename)
        fused_path = os.path.join(fused_folder, filename)
        save_path = os.path.join(save_folder, filename)

        if os.path.exists(debanded_path) and os.path.exists(fused_path):
            generate_comparison_figure(debanded_path, fused_path, save_path)
            print(f"Saved comparison: {save_path}")
        else:
            print(f"Skipping {filename}: file missing.")

if __name__ == "__main__":
    debanded_folder = "../test_img/WaveMamba"
    fused_folder = "../test_img/deepDeband_wavelet"
    comparison_output_folder = "../test_img/comparison_visuals"

    process_comparison_batch(debanded_folder, fused_folder, comparison_output_folder)
    print("All comparison images saved.")
