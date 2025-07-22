import pandas as pd

deband_name = "Banded_Input"
# WaveMamba + wavelet map (separate)
# deband_name = "WaveMamba"
# deband_name = "WaveMamba_wavelet"
# WaveMamba + wavelet map (combined)
# deband_name = "WaveMamba-map"
# deband_name = "WaveMamba-dwt"
# deband_name = 'deepDeband-f'
# deband_name = 'deepDeband-f_wavelet'

# deband_name = "other"
dataset_name = 'HD_images_dataset_dbi'

df = pd.read_csv(f"./combined_metrics/{dataset_name}/combined_metrics_{deband_name}.csv")
output_csv = f"./combined_metrics/{dataset_name}/statistic_metrics_{deband_name}.csv"


metrics = ["PSNR", "SSIM", "LPIPS", "CAMBI", "BBAND" , "DBI"]
# metrics = ["PSNR", "SSIM", "LPIPS", "CAMBI", "BBAND", "CLIP-IQA (quality)", "CLIP-IQA (noisiness)", "CGVQM", "DBI"]
statistics = {
    "Statistic": ["Mean", "Median", "Min", "Max", "Std"]
}

for metric in metrics:
    mean = df[metric].mean()
    median = df[metric].median()
    min_val = df[metric].min()
    max_val = df[metric].max()
    std_val = df[metric].std()
    statistics[metric] = [mean, median, min_val, max_val, std_val]

stats_df = pd.DataFrame(statistics)
print(stats_df)
stats_df.to_csv(output_csv, index=False)