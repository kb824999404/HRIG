import os,platform
import argparse
import shutil
from tqdm import tqdm
import json
import random

import ipdb

def get_args():
    parser = argparse.ArgumentParser(description='Create Dataset')

    # 背景图像目录
    parser.add_argument('-b', '--background',
                    type=str,
                    help='Background images path')
                    
    # 雨层图像目录
    parser.add_argument('-r', '--rain',
                    type=str,
                    help='Rain layer images path')

    # 输出目录
    parser.add_argument('-o', '--output',
                    type=str,
                    help='Output images path')
    # 数据集增强比例
    parser.add_argument('-n', '--number_ratio',
                    type=int,
                    default=1,
                    help='Data augementation ratio')

    args = parser.parse_args()
    return args

def create_dir_not_exist(path):
    sys = platform.system()
    if sys == "Windows":
        for length in range(0, len(path.split(os.path.sep))):
            check_path = os.path.sep.join(path.split(os.path.sep)[:(length+1)])
            if not os.path.exists(check_path):
                os.mkdir(check_path)
                print(f'Created Dir: {check_path}')
    elif sys == "Linux":
        if not os.path.exists(path):
            os.system(f'mkdir -p {path}')
            print(f'Created Dir: {path}')

def copyDir(src,target):
    create_dir_not_exist(target)
    if os.path.exists(src):
        shutil.copytree( src,target,dirs_exist_ok=True)
        print(f"Copying {src} to {target}")

def getDataset():
    samples = []

    return samples
                
if __name__=="__main__":
    args = get_args()
    backgroundRoot = args.background
    rainRoot = args.rain
    outputRoot = args.output
    create_dir_not_exist(args.output)

    useIntensitys = [ i for i in range(10,201,10) ]
    # useIntensitys = [ i for i in range(10,101,10) ]
    # useIntensitys = [ i for i in range(100,201,10) ]

    useWinds = [ "wind_"+w for w in "-0.5 -0.4 -0.3 -0.2 -0.1 0 0.5 0.4 0.3 0.2 0.1".split(' ') ]

    copyDir(backgroundRoot,os.path.join(outputRoot,"background"))

    resultItems = []
    backgroundFiles = os.listdir(backgroundRoot)
    winds =[ item for item in os.listdir(rainRoot) if os.path.isdir(os.path.join(rainRoot,item)) ]
    allRains = []
    for wind in winds:
        if wind not in useWinds:
            continue
        intensitys = os.listdir(os.path.join(rainRoot,wind))
        for intensity in intensitys:
            intensityValue = int(intensity[:-2])
            if intensityValue in useIntensitys:
                copyDir(os.path.join(rainRoot,wind,intensity),os.path.join(outputRoot,"rain",wind,intensity))
                frames = os.listdir(os.path.join(rainRoot,wind,intensity,"rainy_image"))
                for frame in frames:
                    allRains.append({
                        "wind": wind,
                        "intensity": intensityValue,
                        "rain_mask": os.path.join("rain",wind,intensity,"rain_mask",frame),
                        "rainy_image": os.path.join("rain",wind,intensity,"rainy_image",frame)
                    })

    for background in backgroundFiles:
        selectedRains = random.sample(allRains,k=args.number_ratio)
        for rain in selectedRains:
            resultItems.append({
                "wind": rain["wind"],
                "intensity": rain["intensity"],
                "background": os.path.join("background",background),
                "rain_mask": rain["rain_mask"],
                "rainy_image": rain["rainy_image"]
            })
                        
    print("Background Count:",len(backgroundFiles))
    print("Items Count:",len(resultItems))
    with open(os.path.join(outputRoot,"dataset.json"),"w") as f:
        json.dump(resultItems,f)
        print("Writing dataset to ",os.path.join(outputRoot,"dataset.json"))