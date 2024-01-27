dataset_root=/home/ubuntu/data/SyntheticRain_Augmentation
output_root=/home/ubuntu/data/SyntheticRain_Augmentation_Folders

# RainTrainL no augmentation
name=RainTrainL
python get_dataset_folders.py -d ${dataset_root}/${name} -o ${output_root}/${name}

# RainTrainL ratio1
name=RainTrainL_ratio1
python get_dataset_folders.py -d ${dataset_root}/${name} -o ${output_root}/${name}

# RainTrainH no augmentation
name=RainTrainH
python get_dataset_folders.py -d ${dataset_root}/${name} -o ${output_root}/${name}

# RainTrainH ratio1
name=RainTrainH_ratio1
python get_dataset_folders.py -d ${dataset_root}/${name} -o ${output_root}/${name}

# Rain12600 no augmentation
name=Rain12600
python get_dataset_folders.py -d ${dataset_root}/${name} -o ${output_root}/${name}

# Rain12600 ratio1
name=Rain12600_ratio1
python get_dataset_folders.py -d ${dataset_root}/${name} -o ${output_root}/${name}