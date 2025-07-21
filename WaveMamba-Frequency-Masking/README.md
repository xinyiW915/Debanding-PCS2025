# WaveMamba-Frequency-Masking

## Dependencies and Installation

### System Requirements

- Ubuntu ≥ 22.04
- CUDA ≥ 11.8
- Python ≥ 3.8
- PyTorch ≥ 2.0.1

### Setup Steps

```bash
# Clone the repository or navigate to the project root
cd WaveMamba-Frequency-Masking

# Create and activate a new conda environment
conda create -n wavemamba-masking python=3.8
conda activate wavemamba-masking

# Install Python dependencies
pip install -r requirements.txt

# (Recommended) Install the project in editable mode
pip install -e .
````

## Project Structure

```
WaveMamba-Frequency-Masking/
├── basicsr/               # Core code (models, archs, utils, train/test)
│   ├── archs/             # Network architecture
│   ├── models/            # Model wrappers
│   ├── data/              # Dataset preprocessing
│   ├── train.py           # Training entry
│   ├── test.py            # Testing entry
│   └── utils/             # Misc tools
├── options/               # YAML configs for training
├── inference_wavemamba.py # Inference script for WaveMamba
├── inference.py           # Inference script for WaveMamba Variants
├── requirements.txt       # Dependencies
├── setup.py               # Project setup file
```

## Usage

### WaveMamba Variants

#### 🔹 WaveMamba-WWM
After obtaining the debanded output from WaveMamba inference:
```bash
python Debanding_PCS2025/src/wavelet_deband_fusion.py
```

#### 🔹 WaveMamba-DWT  
In `basicsr/archs/wavemamba_arch.py`, set:
```python
class UNet(nn.Module):
    def __init__(..., use_dwt_weight=True):
        ...
        self.use_dwt_weight = use_dwt_weight
```
#### 🔹 WaveMamba-MAP  
In `basicsr/archs/wavemamba_arch.py`, set:
```python
class UNet(nn.Module):
    def __init__(..., use_dwt_weight=False):
        ...
        self.use_dwt_weight = use_dwt_weight
```

### Train

```bash
CUDA_VISIBLE_DEVICES=0 python ./basicsr/train.py -opt ./options/train_deepDeband_dataset.yml
```

### Inference

```bash
python inference.py -i path/to/banded_image.png -g path/to/reference_image.png -w ./ckpt/net_g_latest_WaveMamba_map.pth -o path/to/output/

python inference_wavemamba.py -i path/to/banded_image.png -g path/to/reference_image.pn -w ./ckpt/net_g_latest_WaveMamba.pth -o path/to/output/
```

### Acknowledgements

Based on [BasicSR](https://github.com/XPixelGroup/BasicSR) and [Wave-Mamba](https://github.com/AlexZou14/Wave-Mamba) framework.