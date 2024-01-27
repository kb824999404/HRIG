pred_root=/home/ubuntu/Code/HRIG/logs/2023-10-21T21-50-57_blender-hrig-rainlayer+masked-gdm512-hw512-hybrid-unet128-em3
dataset_root=/home/ubuntu/data/RealBGMaskNew
output_root=/home/ubuntu/data/SyntheticRain_Augmentation_New_Lighten

# # RainTrainL
# origin_dataset_path=/home/ubuntu/data/SyntheticRain/RainTrainL

# # RainTrainL no augmentation
# output_path=${output_root}/RainTrainL
# # python get_augmentation_dataset.py -od ${origin_dataset_path} -o ${output_path} --no_augmentation True

# RainTrainL ratio1
# output_path=${output_root}/RainTrainL_ratio1
# pred_path=${pred_root}/results-real-epoch=000098-RainTrainL_ratio1
# dataset_path=${dataset_root}/RainTrainL_rain512_randomRain_ratio1
# python get_augmentation_dataset.py --pred ${pred_path} -d ${dataset_path} -od ${origin_dataset_path} -o ${output_path}


# # RainTrainH
# origin_dataset_path=/home/ubuntu/data/SyntheticRain/RainTrainH

# # RainTrainH no augmentation
# output_path=${output_root}/RainTrainH
# # python get_augmentation_dataset.py -od ${origin_dataset_path} -o ${output_path} --no_augmentation True

# RainTrainH ratio1
# output_path=${output_root}/RainTrainH_ratio1
# pred_path=${pred_root}/results-real-epoch=000098-RainTrainH_ratio1
# dataset_path=${dataset_root}/RainTrainH_rain512_randomRain_ratio1
# python get_augmentation_dataset.py --pred ${pred_path} -d ${dataset_path} -od ${origin_dataset_path} -o ${output_path}


# Rain12600
# origin_dataset_path=/home/ubuntu/data/SyntheticRain/Rain12600

# Rain12600 no augmentation
# output_path=${output_root}/Rain12600
# python get_augmentation_dataset.py -od ${origin_dataset_path} -o ${output_path} --no_augmentation True

# Rain12600 ratio1
# output_path=${output_root}/Rain12600_ratio1
# pred_path=${pred_root}/results-real-epoch=000098-Rain12600_ratio1
# dataset_path=${dataset_root}/Rain12600_rain512_randomRain_ratio1
# python get_augmentation_dataset.py --pred ${pred_path} -d ${dataset_path} -od ${origin_dataset_path} -o ${output_path}

# Rain1400
# origin_dataset_path=/home/ubuntu/data/SyntheticRain/Rain1400

# Rain1400 no augmentation
# output_path=${output_root}/Rain1400
# python get_augmentation_dataset.py -od ${origin_dataset_path} -o ${output_path} --no_augmentation True

# Rain1400 ratio1
# output_path=${output_root}/Rain1400_ratio1
# pred_path=${pred_root}/results-real-epoch=000098-Rain1400_ratio1
# dataset_path=${dataset_root}/Rain1400_rain512_randomRain_ratio1
# python get_augmentation_dataset.py --pred ${pred_path} -d ${dataset_path} -od ${origin_dataset_path} -o ${output_path}


# SPA-Data
origin_dataset_path=/home/ubuntu/data/SPA-Data

# # SPA-Data no augmentation
# output_path=${output_root}/SPAData
# python get_augmentation_dataset.py -od ${origin_dataset_path} -o ${output_path} --no_augmentation True

# # SPA-Data ratio1
output_path=${output_root}/SPAData_ratio1
pred_path=${pred_root}/results-real-epoch=000098-SPAData_ratio1
dataset_path=${dataset_root}/SPAData_rain512_randomRain_ratio1
python get_augmentation_dataset.py --pred ${pred_path} -d ${dataset_path} -od ${origin_dataset_path} -o ${output_path}

# MSPFN
# origin_dataset_path=/home/ubuntu/data/mixtrain

# output_path=${output_root}/MSPFN
# python get_augmentation_dataset.py -od ${origin_dataset_path} -o ${output_path} --no_augmentation True

# output_path=${output_root}/MSPFN_ratio1
# pred_path=${pred_root}/results-real-epoch=000098-MSPFN_ratio1
# dataset_path=${dataset_root}/MSPFN_rain512_randomRain_ratio1
# python get_augmentation_dataset.py --pred ${pred_path} -d ${dataset_path} -od ${origin_dataset_path} -o ${output_path}