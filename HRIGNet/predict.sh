# LDM
#python predict.py -r logs/2023-10-20T18-52-59_blender-ldm-masked-hw512-concat-unet128-em3

# HRIG
# python predict.py -r logs/2023-03-16T13-08-29_hrig-f16-em3-unet32-hw512

# Real Crop
# python predict_real.py -r logs/2023-10-21T21-50-57_blender-hrig-rainlayer+masked-gdm512-hw512-hybrid-unet128-em3 -c "epoch=000098" -p "predict.yaml" --steps 20

# Real Resize
# python predict_real.py -r logs/2023-10-21T21-50-57_blender-hrig-rainlayer+masked-gdm512-hw512-hybrid-unet128-em3 -c "epoch=000098" -p "predict_R100L_ratio1.yaml" --steps 20 --useResize True --pred_name R100L_ratio1
# python predict_real.py -r logs/2023-10-21T21-50-57_blender-hrig-rainlayer+masked-gdm512-hw512-hybrid-unet128-em3 -c "epoch=000098" -p "predict_R100H_ratio1.yaml" --steps 20 --useResize True --pred_name R100H_ratio1
# python predict_real.py -r logs/2023-10-21T21-50-57_blender-hrig-rainlayer+masked-gdm512-hw512-hybrid-unet128-em3 -c "epoch=000098" -p "predict_R100L_ratio2.yaml" --steps 20 --useResize True --pred_name R100L_ratio2
# python predict_real.py -r logs/2023-10-21T21-50-57_blender-hrig-rainlayer+masked-gdm512-hw512-hybrid-unet128-em3 -c "epoch=000098" -p "predict_R100H_ratio2.yaml" --steps 20 --useResize True --pred_name R100H_ratio2

# python predict_real.py -r logs/2023-10-21T21-50-57_blender-hrig-rainlayer+masked-gdm512-hw512-hybrid-unet128-em3 -c "epoch=000098" -p "predict_RainTrainL_ratio1.yaml" --steps 20 --useResize True --pred_name RainTrainL_ratio1 --gpu 1
# python predict_real.py -r logs/2023-10-21T21-50-57_blender-hrig-rainlayer+masked-gdm512-hw512-hybrid-unet128-em3 -c "epoch=000098" -p "predict_RainTrainH_ratio1.yaml" --steps 20 --useResize True --pred_name RainTrainH_ratio1 --gpu 1
# python predict_real.py -r logs/2023-10-21T21-50-57_blender-hrig-rainlayer+masked-gdm512-hw512-hybrid-unet128-em3 -c "epoch=000098" -p "predict_Rain12600_ratio1.yaml" --steps 20 --useResize True --pred_name Rain12600_ratio1 --gpu 1
# python predict_real.py -r logs/2023-10-21T21-50-57_blender-hrig-rainlayer+masked-gdm512-hw512-hybrid-unet128-em3 -c "epoch=000098" -p "predict_RainTrainL_ratio2.yaml" --steps 20 --useResize True --pred_name RainTrainL_ratio2 --gpu 1
# python predict_real.py -r logs/2023-10-21T21-50-57_blender-hrig-rainlayer+masked-gdm512-hw512-hybrid-unet128-em3 -c "epoch=000098" -p "predict_RainTrainH_ratio2.yaml" --steps 20 --useResize True --pred_name RainTrainH_ratio2 --gpu 1
# python predict_real.py -r logs/2023-10-21T21-50-57_blender-hrig-rainlayer+masked-gdm512-hw512-hybrid-unet128-em3 -c "epoch=000098" -p "predict_Rain12600_ratio2.yaml" --steps 20 --useResize True --pred_name Rain12600_ratio2 --gpu 1

#### Blender ####
# LDM
# python predict.py -r logs/2023-10-20T18-52-59_blender-ldm-masked-hw512-concat-unet128-em3 -c "epoch=000099" --steps 20 -bs 4 --gpu 1
# DiT
# python predict.py -r logs/2023-10-22T16-11-42_blender-dit_b_2-em3-transformer128-hw512-concat -c "epoch=000099" --steps 20 -bs 4 --gpu 1
# HRIG
# python predict.py -r logs/2023-10-21T21-50-57_blender-hrig-rainlayer+masked-gdm512-hw512-hybrid-unet128-em3 -c "epoch=000098" --steps 20 -bs 4 --gpu 1
# GDM Rain Layer 256
# python predict.py -r logs/2023-10-27T15-46-19_blender-hrig-rainlayer+masked-gdm-hw512-hybrid-unet128-em3 -c "epoch=000098" --steps 20 -bs 4 --gpu 1 --guiding True
# GDM Rainy 256
# python predict.py -r logs/2023-11-01T18-52-36_blender-hrig-rainy+masked-gdm-hw512-hybrid-unet128-em3 -c "epoch=000098" --steps 20 -bs 4 --gpu 1 --guiding True
# HRIG DiT
python predict.py -r logs/2023-11-14T10-38-05_blender-hrig-rainlayer+masked-gdm512-transformer-hw512-hybrid-unet128-em3 -c "epoch=000075" --steps 20 -bs 4 --gpu 1
