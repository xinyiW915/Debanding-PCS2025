python inference.py -i /home/xinyi/Project/Debanding/test_img/banded -g /home/xinyi/Project/Debanding/test_img/pristine -w ./ckpt/net_g_latest_WaveMamba_dwt.pth -o /home/xinyi/Project/Debanding/test_img/debanded
python inference.py -i /media/on23019/server1/video_dataset/debanding_dataset/BAND-2k/Image_source/quantified_png -g /media/on23019/server1/video_dataset/debanding_dataset/BAND-2k/Image_source/Origin_image -w ./ckpt/net_g_latest_WaveMamba_dwt.pth -o /home/xinyi/Project/Debanding/WaveMamba-dwt/test_results/BAND-2k/WaveMamba-dwt
python inference.py -i /media/on23019/server1/video_dataset/debanding_dataset/BAND-2k/Image_source/quantified_png -g /media/on23019/server1/video_dataset/debanding_dataset/BAND-2k/Image_source/Origin_image -w ./ckpt/net_g_latest_WaveMamba_map.pth -o /home/xinyi/Project/Debanding/WaveMamba-dwt/test_results/BAND-2k/WaveMamba-map
python inference.py -i /media/xinyi/server/video_dataset/debanding_dataset/HD_images_dataset_dbi/Quantized_Images -g /media/xinyi/server/video_dataset/debanding_dataset/HD_images_dataset_dbi/Original_Images -w ./ckpt/net_g_latest_WaveMamba_map.pth -o /home/xinyi/Project/Debanding/WaveMamba-dwt/test_results/HD_images_dataset_dbi/WaveMamba-map
python inference.py -i /media/xinyi/server/video_dataset/debanding_dataset/HD_images_dataset_dbi/Quantized_Images -g /media/xinyi/server/video_dataset/debanding_dataset/HD_images_dataset_dbi/Original_Images -w ./ckpt/net_g_latest_WaveMamba_dwt.pth -o /home/xinyi/Project/Debanding/WaveMamba-dwt/test_results/HD_images_dataset_dbi/WaveMamba_dwt



python inference_wavemamba.py -i /media/on23019/server1/video_dataset/debanding_dataset/BAND-2k/Image_source/quantified_png -g /media/on23019/server1/video_dataset/debanding_dataset/BAND-2k/Image_source/Origin_image -w ./ckpt/net_g_latest_WaveMamba.pth -o /home/xinyi/Project/Debanding/Wave-Mamba/results/BAND-2k/WaveMamba
python inference_wavemamba.py -i /media/xinyi/server/video_dataset/debanding_dataset/HD_images_dataset_dbi/Quantized_Images -g /media/xinyi/server/video_dataset/debanding_dataset/HD_images_dataset_dbi/Original_Images -w ./ckpt/net_g_latest_WaveMamba.pth -o /home/xinyi/Project/Debanding/Wave-Mamba/results/HD_images_dataset_dbi/WaveMamba


