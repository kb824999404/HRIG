import os
import time

def clean(models):
    for model in models:
        ckptsPath = os.path.join(logPath,model,"checkpoints")
        ckpts = os.listdir(ckptsPath)
        ckpts.sort()
        for ckpt in ckpts[:-2]:
            print(ckpt)
            if ckpt[:1] == "e":
                os.remove(os.path.join(ckptsPath,ckpt))
                print("Remove "+os.path.join(ckptsPath,ckpt))

if __name__=="__main__":
    logPath = "logs"
    models = os.listdir(logPath)
    while True:
        clean(models)
        print("Clean!")
        time.sleep(60)