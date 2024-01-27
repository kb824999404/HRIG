# LDM
python main.py --base configs/ldm_blender/blender-ldm-masked-hw512-concat-unet128-em3.yaml -t --gpus 0,  

# GDM 256 Rain Layer
python main.py --base configs/hrig_blender/blender-gdm-rainlayer-hw256-f4-em3.yaml -t --gpus 1,
python main.py --base configs/hrig_blender/blender-hrig-rainlayer+masked-gdm-hw512-hybrid-unet128-em3.yaml -t --gpus 1,  

# GDM 256 Rainy
python main.py --base configs/hrig_blender/blender-gdm-rainy-hw256-f4-em3.yaml -t --gpus 0,
python main.py --base configs/hrig_blender/blender-hrig-rainy+masked-gdm-hw512-hybrid-unet128-em3.yaml -t --gpus 0,  

# GDM 512 Rain Layer
python main.py --base configs/hrig_blender/blender-gdm-rainlayer-hw512-f4-em3.yaml -t --gpus 1,
python main.py --base configs/hrig_blender/blender-hrig-rainlayer+masked-gdm512-hw512-hybrid-unet128-em3.yaml -t --gpus 1,


# DiT
python main.py --base configs/dit_blender/blender-dit_b_2-em3-transformer128-hw512-concat.yaml -t --gpus 1,

# DiT GDM 512 Rain Layer
python main.py --base configs/dit_blender/blender-gdm-dit_b_2-em3-transformer128-hw512-concat.yaml -t --gpus 0,
python main.py --base configs/dit_blender/blender-hrig-rainlayer+masked-gdm512-transformer-hw512-hybrid-unet128-em3.yaml -t --gpus 1,