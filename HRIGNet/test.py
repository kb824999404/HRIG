import argparse, os, sys, datetime
from omegaconf import OmegaConf
from main import instantiate_from_config
import numpy as np
from PIL import Image
from tqdm import tqdm
import ipdb

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="name of config",
    )


    return parser

def showData(sample):
    Image.fromarray(sample["raw_background"]).save(os.path.join(logPath,"raw_background.jpg"))
    Image.fromarray(sample["raw_rainy_image"]).save(os.path.join(logPath,"raw_rainy_image.jpg"))
    Image.fromarray(sample["raw_rain_layer"]).save(os.path.join(logPath,"raw_rain_layer.png"))
    Image.fromarray(sample["raw_rain_layer"][...,:3]).save(os.path.join(logPath,"rain_layer.jpg"))
    mask = sample["mask"].astype(np.uint8) * 255
    Image.fromarray(mask,mode='L').save(os.path.join(logPath,"mask.jpg"))

    masked_background = np.clip((sample["masked_background"] + 1.0) * 127.5,0,255).astype(np.uint8)
    Image.fromarray(masked_background).save(os.path.join(logPath,"masked_background.jpg"))


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    config = OmegaConf.load(opt.config)

    # data
    config.data.params.train.params.fullSize = True
    config.data.params.validation.params.fullSize = True
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    trainDataset = data.datasets['train']
    testDataset = data.datasets['validation']

    logPath = os.path.join("logs","testOutput")
    os.makedirs(logPath, exist_ok=True)

    for index,sample in enumerate(trainDataset.labels):
        print(index)
        if sample["rain_layer_path"] == "/home/ubuntu/data/BlenderRain/rainy/lane/front/100mm/wind_0_0_0_0/rain_layer/frame_0910.png":
            print("Fonud!")
            showData(trainDataset[index])
            break
    