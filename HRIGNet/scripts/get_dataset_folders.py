import os,platform,sys
import argparse
import shutil
from tqdm import tqdm
import json
import random

import glob2


def get_args():
    parser = argparse.ArgumentParser(description='Create Dataset')


    # 增强数据集目录
    parser.add_argument('-d', '--dataset',
                    type=str,
                    help='Dataset path')

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

    with open(os.path.join(args.dataset,"dataset.json"),"r") as f:
        dataset = json.load(f)
    
    input_root = os.path.join(args.output,"input")
    label_root = os.path.join(args.output,"label")
    create_dir_not_exist(input_root)
    create_dir_not_exist(label_root)
    
    for index in tqdm(range(len(dataset))):
        sample = dataset[index]
        postfix = sample["input"][-4:]
        shutil.copyfile(os.path.join(args.dataset,sample["input"]),os.path.join(input_root,str(index)+postfix))
        shutil.copyfile(os.path.join(args.dataset,sample["label"]),os.path.join(label_root,str(index)+postfix))