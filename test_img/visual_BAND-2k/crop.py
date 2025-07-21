import os
from PIL import Image

# ==== 参数配置 ====
input_root = "./"
output_root = "./patches_output"
gt_folder_name = "Pristine"
method_folders = ["Banded", "WaveMamba", "WaveMamba-WWM", "WaveMamba-dwt", "WaveMamba-map"]
patch_w, patch_h = 320, 360  # 可整除 1920x1080
image_w, image_h = 1920, 1080

os.makedirs(output_root, exist_ok=True)

# 获取图像名
gt_folder = os.path.join(input_root, gt_folder_name)
image_list = sorted([f for f in os.listdir(gt_folder) if f.endswith(('.png', '.jpg'))])

# 计算 patch 位置
x_steps = image_w // patch_w
y_steps = image_h // patch_h

# ==== 主处理 ====
for image_name in image_list:
    image_id = os.path.splitext(image_name)[0]
    out_dir = os.path.join(output_root, image_id)
    os.makedirs(out_dir, exist_ok=True)

    # 加载所有图像（GT + 方法）
    images = {}
    for method in method_folders + [gt_folder_name]:
        path = os.path.join(input_root, method, image_name)
        if os.path.exists(path):
            images[method] = Image.open(path).convert("RGB")

    # 裁剪每个 patch 坐标
    patch_idx = 1
    for yi in range(y_steps):
        for xi in range(x_steps):
            x0 = xi * patch_w
            y0 = yi * patch_h
            x1 = x0 + patch_w
            y1 = y0 + patch_h

            for method, img in images.items():
                patch = img.crop((x0, y0, x1, y1))
                label = "GroundTruth" if method == gt_folder_name else method
                patch.save(os.path.join(out_dir, f"patch_{patch_idx}_{label}.png"))

            patch_idx += 1

print("done")
