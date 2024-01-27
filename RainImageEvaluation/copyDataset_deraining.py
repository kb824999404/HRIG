import os,sys
import argparse
import json
from PIL import Image
import numpy as np
import shutil

from tqdm import tqdm
import ipdb


def check_arg(args):
    parser = argparse.ArgumentParser(description='Copy Dataset')

    parser.add_argument('-d','--dataset',
                        help='Dataset Root',
                        type=str, required=True)

    parser.add_argument('-n','--name',
                        help='Dataset Name',
                        type=str, required=True)
    
    parser.add_argument('-o','--output',
                        help='Output Root',
                        type=str, required=True)

    
    return parser.parse_args(args)

def create_dir_not_exist(path):
    for length in range(0, len(path.split(os.path.sep))):
        check_path = os.path.sep.join(path.split(os.path.sep)[:(length+1)])
        if not os.path.exists(check_path):
            os.mkdir(check_path)
            print(f'Created Dir: {check_path}')

if __name__=="__main__":
    args = check_arg(sys.argv[1:])
    datasetPath = args.dataset
    datasetName = args.name


    outputPath = args.output
    
    create_dir_not_exist(outputPath)
    if datasetName in ["Rain100L","Rain100H"]:
        for sample in tqdm(os.listdir(os.path.join(datasetPath,"norain"))):
            shutil.copy(os.path.join(datasetPath,"norain",sample),os.path.join(outputPath,sample[2:]))
    elif datasetName in ["Rain1400"]:
        for sample in tqdm(os.listdir(os.path.join(datasetPath,"rainy_image"))):
            shutil.copy(os.path.join(datasetPath,"ground_truth",sample.split('_')[0]+'.jpg'),os.path.join(outputPath,sample))
    elif datasetName in ["SPAData"]:
        for sample in tqdm(os.listdir(os.path.join(datasetPath,"rain-origin"))):
            shutil.copy(os.path.join(datasetPath,"label",sample.split('.')[0]+'gt.png'),os.path.join(outputPath,sample))