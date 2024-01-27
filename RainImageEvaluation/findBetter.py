import os
from PIL import Image
import numpy as np
from pytorch_fid import fid_score
from piqa import LPIPS,SSIM,PSNR
import torch
import json
from tqdm import tqdm

def loadImg(path,size=None):
    image = Image.open(path).convert("RGB")

    image = np.array(image).astype(np.uint8)

    image = (image / 255.0).astype(np.float32)
    image = image.transpose(2,0,1)
    return image


if __name__=="__main__":
    Device = torch.device("cuda")

    # predRoot = r"F:\Experiments\SIRR\PReNet_result"
    predRoot = r"F:\Experiments\SIRR\Restormer_result"
    # predRoot = r"F:\Experiments\SIRR\SFNet_result"
    # predRoot = r"F:\Experiments\SIRR\M3SNet_result"
    datasets = [ "Rain1400","RainTrainL" ]
    for dataset in datasets:
        # experiment1 = os.path.join(predRoot,dataset,"SPAData")
        # experiment2 = os.path.join(predRoot,dataset+"_ratio1","SPAData")
        experiment1 = os.path.join(predRoot,dataset+"_SPAData")
        experiment2 = os.path.join(predRoot,dataset+"_ratio1"+"_SPAData")

        criterion = PSNR()
        improves = []
        for file in tqdm(os.listdir(experiment1)):
            img1 = loadImg(os.path.join(experiment1,file))
            img2 = loadImg(os.path.join(experiment2,file))

            imgOrigin = loadImg(os.path.join(r"F:\Experiments\SIRR\SPAData-bg",file.split('.')[0]+"gt.png"))
            img1 = torch.tensor(img1).to(device=Device).unsqueeze(0)
            img2 = torch.tensor(img2).to(device=Device).unsqueeze(0)
            imgOrigin = torch.tensor(imgOrigin).to(device=Device).unsqueeze(0)

            loss1 = criterion(img1,imgOrigin)
            loss2 = criterion(img2,imgOrigin)
            improve = (loss2 - loss1).item()
            improves.append((file,improve))
        improves.sort(key=lambda x:x[1],reverse=True)
        with open(os.path.join(predRoot,dataset+"_improves_PSNR.json"),"w") as f:
            json.dump(improves,f,indent=4)
