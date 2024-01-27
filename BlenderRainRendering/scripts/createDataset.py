import os,platform
import argparse
import shutil
from natsort import natsorted
from tqdm import tqdm
import json

def get_args():
    parser = argparse.ArgumentParser(description='Create Dataset')

    # 输入目录
    parser.add_argument('-D', '--data',
                    type=str,
                    default='../data',
                    help='The source data directory')
                    
    # 输出目录
    parser.add_argument('-O', '--output',
                    type=str,
                    default='../data/particles',
                    help='The output directory')

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

def getDataset(scenes):
    samples = []
    for scene,sequences in scenes.items():
        for sequence in sequences:
            print("\n" + "="*40 + f" {scene}-{sequence} " +"="*40)
            bg = os.path.join(scene,sequence,"background")
            depth = os.path.join(scene,sequence,"depth")

            bg_path = os.path.join(args.data,"source",bg)
            depth_path = os.path.join(args.data,"source",depth)
            rain_path = os.path.join(args.data,"output",scene,sequence)
            if not os.path.exists(bg_path):
                print(f"{scene} {sequence} missing background images: {bg_path}")
                continue
            if not os.path.exists(depth_path):
                print(f"{scene} {sequence} missing depth images: {depth_path}")
                continue
            if not os.path.exists(rain_path):
                print(f"{scene} {sequence} missing rainy images: {rain_path}")
                continue
            print(f"{scene} {sequence} is valid")

            # Copy clean images
            copyDir(bg_path, os.path.join(args.output,"clean",bg))
            copyDir(depth_path, os.path.join(args.output,"clean",depth))

            frames = natsorted([ img.split('.')[0] for img in os.listdir(bg_path) ])

            intensities = [  I for I in os.listdir(rain_path) if os.path.isdir(os.path.join(rain_path,I))  ]
            for intensity in intensities:
                intensity_path = os.path.join(rain_path,intensity)
                winds = [  name for name in os.listdir(intensity_path) if os.path.isdir(os.path.join(intensity_path,name))  ]
                for wind in winds:
                    wind_path = os.path.join(scene,sequence,intensity,wind)
                    rain_layer = os.path.join(wind_path,"rain_layer")
                    rainy_depth = os.path.join(wind_path,"depth")
                    rainy_image = os.path.join(wind_path,"rainy_image")

                    print(f"Processing {scene} {sequence} {intensity} {wind}")
                    # Create Samples
                    for frame in frames:
                        sample = {
                            "scene": scene,
                            "sequence": sequence,
                            "intensity": int(intensity[:-2]),
                            "wind": wind,
                            "background": os.path.join("clean",bg,f"{frame}.jpg"),
                            "depth": os.path.join("clean",depth,f"{frame}.png"),
                            "rain_layer": os.path.join("rainy",rain_layer,f"{frame}.png"),
                            "rainy_depth": os.path.join("rainy",rainy_depth,f"{frame}.png"),
                            "rainy_image": os.path.join("rainy",rainy_image,f"{frame}.png")
                        }
                        samples.append(sample)
                    
                    # Copy rainy images
                    for item in [ rain_layer,rainy_depth,rainy_image ]:
                        if len(os.listdir(os.path.join(args.data,"output",item))) == len(frames):
                            copyDir(os.path.join(args.data,"output",item), os.path.join(args.output,"rainy",item))
    return samples
                
                    

if __name__=="__main__":
    args = get_args()
    scenes_train = {
        "lane": [ "front", "mid", "side" ],
        "citystreet": [ "far", "front", "back", "sideleft", "sideright" ]
    }
    scenes_test = {
        "lane": [ "low" ],
        "citystreet": [ "sideinner" ]
    }

    create_dir_not_exist(args.output)

    # Create Trainset
    samples_train = getDataset(scenes_train)
    with open(os.path.join(args.output,"trainset.json"),"w") as f:
        json.dump(samples_train,f)

    # Create Testset
    samples_test = getDataset(scenes_test)
    with open(os.path.join(args.output,"testset.json"),"w") as f:
        json.dump(samples_test,f)

