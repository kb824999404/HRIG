import os,platform,sys
import argparse
import shutil
from tqdm import tqdm
import json
import random

import PIL
from PIL import Image
import numpy as np

import glob2

import ipdb

def get_args():
    parser = argparse.ArgumentParser(description='Create Dataset')

    # 背景图像目录
    parser.add_argument('--pred',
                    type=str,
                    help='Predicted images path')
                    
    # 增强数据集目录
    parser.add_argument('-d', '--dataset',
                    type=str,
                    help='Dataset path')
    # 原始数据集目录
    parser.add_argument('-od', '--origin_dataset',
                    type=str,
                    help='Origin Dataset path')

    # 输出目录
    parser.add_argument('-o', '--output',
                    type=str,
                    help='Output images path')
    # 不使用数据增强
    parser.add_argument('--no_augmentation',
                type=str,
                default="False",
                help='No use data augmentation')

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

    fulldataset = []

    # 获取原始数据集并迁移至新文件夹
    # For Rain100
    # copyDir(os.path.join(args.origin_dataset,"label"),os.path.join(args.output,"label"))
    # copyDir(os.path.join(args.origin_dataset,"input"),os.path.join(args.output,"rain-origin"))

    # for bgFile in os.listdir(os.path.join(args.origin_dataset,"label")):
    #     fulldataset.append({
    #         "label": os.path.join("label",bgFile),
    #         "input": os.path.join("rain-origin","rain-"+bgFile[7:]),
    #     })
    
    # For Rain12600
    # copyDir(os.path.join(args.origin_dataset,"ground_truth"),os.path.join(args.output,"label"))
    # copyDir(os.path.join(args.origin_dataset,"rainy_image"),os.path.join(args.output,"rain-origin"))

    # for rainyFile in os.listdir(os.path.join(args.origin_dataset,"rainy_image")):
    #     fulldataset.append({
    #         "label": os.path.join("label",rainyFile.split("_")[0]+".jpg"),
    #         "input": os.path.join("rain-origin",rainyFile),
    #     })

    # For SPAData
    copyDir(os.path.join(args.origin_dataset,"gt"),os.path.join(args.output,"label"))
    copyDir(os.path.join(args.origin_dataset,"rain"),os.path.join(args.output,"rain-origin"))

    for rainyFile in os.listdir(os.path.join(args.origin_dataset,"rain")):
        fulldataset.append({
            "label": os.path.join("label",rainyFile.split(".")[0]+"gt.png"),
            "input": os.path.join("rain-origin",rainyFile),
        })

    # For MSPFN
    # copyDir(os.path.join(args.origin_dataset,"label"),os.path.join(args.output,"label"))
    # copyDir(os.path.join(args.origin_dataset,"input"),os.path.join(args.output,"rain-origin"))

    # for bgFile in os.listdir(os.path.join(args.origin_dataset,"label")):
    #     fulldataset.append({
    #         "label": os.path.join("label",bgFile),
    #         "input": os.path.join("rain-origin",bgFile),
    #     })


    if args.no_augmentation == "False":
        # 获取雨遮罩数据集信息
        with open(os.path.join(args.dataset,"dataset.json"),"r") as f:
            dataset = json.load(f)
        
        # 获取增强数据集，调整图片尺寸，并迁移至新文件夹
        create_dir_not_exist(os.path.join(args.output,"rain-augmentation"))
        for sample in tqdm(dataset):
            pred_name = "{}_{}_{}_{}.jpg".format(sample["intensity"],sample["wind"],
                                                    sample["background"].split("/")[-1][:-4],  
                                                    sample["rain_mask"].split("/")[-1][:-4])
            label_img = Image.open(os.path.join(args.dataset,sample["background"]))
            pred_img = Image.open(os.path.join(args.pred,pred_name))
            pred_img = pred_img.resize(label_img.size, resample=PIL.Image.BICUBIC)
            mask_img = Image.open(os.path.join(args.dataset,sample["rain_mask"]))
            mask_img = mask_img.resize(label_img.size, resample=PIL.Image.BICUBIC)

            pred_img_lighten = np.max((np.array(label_img),np.array(pred_img)),axis=0)
            mask_img = np.array(mask_img)[...,0:1]
            mask_img_normal = mask_img.copy()
            mask_img_normal[mask_img >= 127] = 1
            mask_img_normal[mask_img < 127] = 0
            label_img = np.array(label_img)

            pred_img_lighten = pred_img_lighten * mask_img_normal + (1-mask_img_normal) * label_img
            pred_img_lighten = Image.fromarray(pred_img_lighten)
            pred_img_lighten.save(os.path.join(args.output,"rain-augmentation",pred_name))
            fulldataset.append({
                "label": os.path.join("label",sample["background"].split("/")[-1]),
                "input": os.path.join("rain-augmentation",pred_name)
            })
            # # For Rain12600
            # fulldataset.append({
            #     "label": os.path.join("label",sample["background"].split("/")[-1].split("_")[0]+".jpg"),
            #     "input": os.path.join("rain-augmentation",pred_name)
            # })

    # 保存结果
    print("Sample Count:",len(fulldataset))
    with open(os.path.join(args.output,"dataset.json"),"w") as f:
        json.dump(fulldataset,f)
