dataset=/home/ubuntu/data/BlenderRain
dataset_resize=data/BlenderRain
predRoot=/home/ubuntu/Code/HRIG/logs/2023-10-20T18-52-59_blender-ldm-masked-hw512-concat-unet128-em3/results-epoch=000099
predName=ldm512
resultRoot=results
python getMetrics.py -d ${dataset} --dataset_resize ${dataset_resize} --predRoot ${predRoot} --predName ${predName} --result ${resultRoot}