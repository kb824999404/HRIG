import os,sys
import argparse
import json
from PIL import Image
import numpy as np

from tqdm import tqdm
import ipdb


def check_arg(args):
    parser = argparse.ArgumentParser(description='Copy Dataset')

    parser.add_argument('-d','--dataset',
                        help='Dataset Root',
                        type=str, required=True)
    
    parser.add_argument('-o','--output',
                        help='Output Root',
                        type=str, required=True)

    parser.add_argument('-iw','--image_width',
                        help='Image Width',
                        type=int, default=2048)
    
    parser.add_argument('-ih','--image_height',
                        help='Image Height',
                        type=int, default=1024)
    
    return parser.parse_args(args)

def create_dir_not_exist(path):
    for length in range(0, len(path.split(os.path.sep))):
        check_path = os.path.sep.join(path.split(os.path.sep)[:(length+1)])
        if not os.path.exists(check_path):
            os.mkdir(check_path)
            print(f'Created Dir: {check_path}')

def loadImg(path,size=None):
    image = Image.open(path).convert("RGB")

    # image = np.array(image).astype(np.uint8)
    # crop = min(image.shape[0], image.shape[1])
    # h, w, = image.shape[0], image.shape[1]
    # image = image[(h - crop) // 2:(h + crop) // 2,
    #         (w - crop) // 2:(w + crop) // 2]
    # image = Image.fromarray(image)

    if size:
        image = image.resize((size[0],size[1]))

    return image


def getDataset(dataset,outputPath):
    for sample in tqdm(dataset):
            gtPath = os.path.join(datasetPath,sample["rainy_image"])
            gtPathOutput = os.path.join(outputPath,"{}_{}_{}_{}_{}.jpg".format(sample["scene"],sample["sequence"],sample["intensity"],
                                 sample["wind"],os.path.split(sample["background"])[-1][:-4]))

            img = loadImg(gtPath,size=imgSize)
            img.save(gtPathOutput)



if __name__=="__main__":
    args = check_arg(sys.argv[1:])
    datasetPath = args.dataset
    trainsetPath = os.path.join(datasetPath,"trainset.json")
    testsetPath = os.path.join(datasetPath,"testset.json")

    imgSize = args.image_width,args.image_height

    outputPathTrain = os.path.join(args.output,"train")
    outputPathTest = os.path.join(args.output,"test")
    
    create_dir_not_exist(outputPathTrain)
    create_dir_not_exist(outputPathTest)


    with open(trainsetPath,"r") as f:
        dataset = json.load(f)
        print("="*20 + "Trainset" + "="*20)
        getDataset(dataset,outputPathTrain)

    with open(testsetPath,"r") as f:
        dataset = json.load(f)
        print("="*20 + "Testset" + "="*20)
        getDataset(dataset,outputPathTest)