import argparse, os, sys, datetime, glob, importlib, csv
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import ipdb
from einops import rearrange

from hrig.data.rain import loadImgRGB

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        type=str,
        const=True,
        default="last",
        nargs="?",
        help="name of ckpt file",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="path of input low-resolution images",
    )
    return parser

# 以masked_image为条件，用随机隐变量生成结果
def predict_from_zero(dataLoader):
    for batch_idx,batch in tqdm(enumerate(dataLoader)):
        batch_size = dataLoader.batch_size

        cond = []
        for path in batch['mask_path']:
            sampleName = os.path.split(path)[-1][:-4]
            inputPath = os.path.join(opt.input,sampleName+"_pred.jpg")
            inputImage = loadImgRGB(inputPath)
            inputImage = np.array(inputImage).astype(np.uint8)
            inputImage = (inputImage / 127.5 - 1.0).astype(np.float32)
            cond.append(inputImage)
        cond = np.array(cond)
        cond = torch.tensor(cond)
        batch["image_1"] = cond

        x_samples_ddim = model.get_samples(batch, batch_size, ddim_steps=opt.steps)
        predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                        min=0.0, max=1.0)
        predicted_image = predicted_image.cpu().numpy().transpose(0,2,3,1)*255

        for path,result in zip(batch['mask_path'],predicted_image):
            path_new = os.path.split(path)[-1][:-4] +"_pred.jpg"
            outpath = os.path.join(outdir,path_new)
            Image.fromarray(result.astype(np.uint8)).save(outpath)

# 以masked_image为条件，用随机隐变量生成结果，结果与Ground Truth通过Mask混合
def predict_from_zero_blend(dataLoader):
    for batch_idx,batch in tqdm(enumerate(dataLoader)):
        image = (batch[model.cond_stage_key]+1.0)/2.0
        if "mask" in batch:
            mask = batch["mask"]
        else:
            mask = batch["mask_0"]
        batch_size = dataLoader.batch_size

        cond = []
        for path in batch['mask_path']:
            sampleName = os.path.split(path)[-1][:-4]
            inputPath = os.path.join(opt.input,sampleName+"_pred.jpg")
            inputImage = loadImgRGB(inputPath)
            inputImage = np.array(inputImage).astype(np.uint8)
            inputImage = (inputImage / 127.5 - 1.0).astype(np.float32)
            cond.append(inputImage)
        cond = np.array(cond)
        cond = torch.tensor(cond)
        batch["image_1"] = cond

        x_samples_ddim = model.get_samples(batch, batch_size, ddim_steps=opt.steps)
        predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                      min=0.0, max=1.0)
        

        predicted_image = predicted_image.cpu().numpy().transpose(0,2,3,1)
        image = image.cpu().numpy()
        mask = mask.unsqueeze(3).cpu().numpy()
        inpainted = (1-mask)*predicted_image+mask*image
        inpainted = inpainted*255


        for path,result in zip(batch['mask_path'],inpainted):
            path_new = os.path.split(path)[-1][:-4] +"_pred.jpg"
            outpath = os.path.join(outdir,path_new)
            Image.fromarray(result.astype(np.uint8)).save(outpath)


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    outdir = os.path.join(logdir,"results-"+opt.ckpt)

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    # model
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(os.path.join(ckptdir,opt.ckpt+".ckpt"))["state_dict"],
                          strict=False)
    print("Restored from "+os.path.join(ckptdir,opt.ckpt+".ckpt"))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    # data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    os.makedirs(outdir, exist_ok=True)

    with torch.no_grad():
        with model.ema_scope():
            trainDataLoader = data._train_dataloader()
            testDataLoader = data._val_dataloader()
            predict_from_zero_blend(trainDataLoader)
            predict_from_zero_blend(testDataLoader)
