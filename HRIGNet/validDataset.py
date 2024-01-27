import json
import os
import argparse
import ipdb

def get_args():
    parser = argparse.ArgumentParser(description='Valid Dataset')

    # 输入目录
    parser.add_argument('-D', '--data',
                    type=str,
                    default=None,
                    help='The dataset directory')


    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    check_keys = [
        "background","depth","rain_layer","rainy_depth","rainy_image"
    ]
    with open(os.path.join(args.data,"trainset.json"),"r") as f:
        trainset = json.load(f)
        print(len(trainset))
        for sample in trainset:
            valid = True
            for key in check_keys:
                if not os.path.exists(os.path.join(args.data,sample[key])):
                    valid = False
                    print("Miss ",key)
                    break
            if not valid:
                print("Miss sample {}-{}-{}-{}-{}".format(sample["scene"],sample["sequence"],sample["intensity"],sample["wind"],sample["background"].split("/")[-1]))
            
                
    with open(os.path.join(args.data,"testset.json"),"r") as f:
        testset = json.load(f)
        print(len(testset))
        for sample in testset:
            valid = True
            for key in check_keys:
                if not os.path.exists(os.path.join(args.data,sample[key])):
                    valid = False
                    print("Miss ",key)
                    break
            if not valid:
                print("Miss sample {}-{}-{}-{}-{}".format(sample["scene"],sample["sequence"],sample["intensity"],sample["wind"],sample["background"].split("/")[-1]))
            
