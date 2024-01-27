import os,sys
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import argparse

from pytorch_fid import fid_score
from piqa import LPIPS,SSIM,PSNR

import warnings
warnings.filterwarnings("ignore")


def check_arg(args):
    parser = argparse.ArgumentParser(description='Get Metrics')

    parser.add_argument('-test','--testset',
                        help='Testset Path',
                        type=str, required=True)

    parser.add_argument('--predRoot', type=str,
                        help='predict result path')

    parser.add_argument('--predName', type=str,
                        help='predict result name')
    
    parser.add_argument('-r','--result',
                        help='Result Root',
                        type=str, required=True)
    
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
    
    # if size:
    #     image = Image.fromarray(image)
    #     image = image.resize((size,size))
    image = np.array(image).astype(np.uint8)

    image = (image / 255.0).astype(np.float32)
    image = image.transpose(2,0,1)
    return image

def getFID():
    print("="*20 + "FID" + "="*20)

    fid = fid_score.calculate_fid_given_paths(paths=[ os.path.join(testsetPath),os.path.join(predRoot) ],
                                        batch_size=1,device=Device,dims=2048,num_workers=0)
    
    return fid


def getLosses(path):
    losses = [0,0,0]
    count = [0,0,0]
    criterions =[LPIPS(),SSIM(),PSNR()]
    for index in range(3):
        criterions[index] = criterions[index].to(device=Device)

    for sample in tqdm(os.listdir(path)):
        gtPath = os.path.join(path,sample)
        predPath = os.path.join(predRoot,sample)

        if not os.path.exists(predPath):
            continue

        img_gt = loadImg(gtPath)
        img_gt = torch.tensor(img_gt).to(device=Device).unsqueeze(0)
        img_pred = loadImg(predPath)
        img_pred = torch.tensor(img_pred).to(device=Device).unsqueeze(0)
        for index in range(3):
            loss = criterions[index](img_gt,img_pred)
            if torch.isnan(loss):
                continue
            losses[index] += loss
            count[index] += 1

    loss_avg = []
    for index in range(3):
        loss_avg.append((losses[index]/count[index]).reshape(-1)[0].cpu().detach().numpy())

    return loss_avg

if __name__=="__main__":
    args = check_arg(sys.argv[1:])

    testsetPath = args.testset

    Device = torch.device("cuda")

    predRoot = args.predRoot
    predName = args.predName

    print("TestsetPath:",testsetPath)
    print("PredPath:",predRoot)

    losses = getLosses(testsetPath)
    lpips_test,ssim_test,psnr_test = losses[0],losses[1],losses[2]

    print("Test:")
    print("\tLPIPS:",lpips_test)
    print("\tSSIM:",ssim_test)
    print("\tPSNR:",psnr_test)

    # fid_train = getFID("train")
    fid_test = getFID()
    print("\tFID:",fid_test)

    result =  {
        "fid_test": float(fid_test),
        "lpips_test": float(lpips_test),
        "ssim_test": float(ssim_test),
        "psnr_test": float(psnr_test),
    }
    create_dir_not_exist(os.path.join(args.result,predName))
    with open(os.path.join(args.result,predName,"result.json"),"w") as f:
        json.dump(result,f)