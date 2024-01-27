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

    # useIntensitys = [ 10,50,100 ]
    useIntensitys = [ 10,20,30,40,50,60,70,80,90,100 ]
    randomBG = True

    copyDir(backgroundRoot,os.path.join(outputRoot,"background"))

    resultItems = []
    backgroundFiles = os.listdir(backgroundRoot)
    winds =[ item for item in os.listdir(rainRoot) if os.path.isdir(os.path.join(rainRoot,item)) ]
    for wind in winds:
        intensitys = os.listdir(os.path.join(rainRoot,wind,"rain"))
        for intensity in intensitys:
            intensityValue = int(intensity[:-2])
            if intensityValue in useIntensitys:
                copyDir(os.path.join(rainRoot,wind,"rain",intensity),os.path.join(outputRoot,"rain",wind,intensity))
                frames = os.listdir(os.path.join(rainRoot,wind,"rain",intensity,"rainy_image"))
                for frame in frames:
                    if randomBG:
                        background = random.sample(backgroundFiles,k=1)[0]
                        resultItems.append({
                            "wind": wind,
                            "intensity": intensityValue,
                            "background": os.path.join("background",background),
                            "rain_mask": os.path.join("rain",wind,intensity,"rain_mask",frame),
                            "rainy_image": os.path.join("rain",wind,intensity,"rainy_image",frame)
                        })
                    else:
                        for background in backgroundFiles:
                            resultItems.append({
                                "wind": wind,
                                "intensity": intensityValue,
                                "background": os.path.join("background",background),
                                "rain_mask": os.path.join("rain",wind,intensity,"rain_mask",frame),
                                "rainy_image": os.path.join("rain",wind,intensity,"rainy_image",frame)
                            })
                        
    print("Items Count:",len(resultItems))
    with open(os.path.join(outputRoot,"dataset.json"),"w") as f:
        json.dump(resultItems,f)
        print("Writing dataset to ",os.path.join(outputRoot,"dataset.json"))