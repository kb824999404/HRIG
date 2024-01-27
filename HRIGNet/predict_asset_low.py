from main_utils import show_output, create_folder, show_img, write_cv2_img_jpg
import os
from omegaconf import OmegaConf
import time
import yaml
from asset.models.cond_transformer import Net2NetTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
from main_utils import laplacian_blend
import albumentations
import random
import argparse
from main import instantiate_from_config

from tqdm import tqdm

import ipdb

def predict(sample):
    # ------ get downsampled inputs
    mask = sample["mask"]
    mask_guide = mask.copy()
    mask_guide = np.clip(mask_guide, 0, 1)
    mask_guide_tensor = torch.from_numpy(mask_guide).to(model.device).unsqueeze(0).unsqueeze(0).to(memory_format=torch.contiguous_format)  # 1, 1, 256, 256, on cuda
    resized_mask_guide_tensor = F.interpolate(mask_guide_tensor, size=(16, 16))
    latent_mask_guide = resized_mask_guide_tensor.squeeze().cpu().numpy()  # [0, 1]

    source_guide = torch.tensor(sample["image"].transpose(2, 0, 1)[None]).to(dtype=torch.float32, device=model.device)  # 1, 3, h, w
    z_code_guide, z_indices_guide = model.encode_to_z(source_guide, mask_tensor=mask_guide_tensor)  # VQGAN encoding

    c_code_guide, c_indices_guide = model.encode_to_c(source_guide)  # 1, 256, 16, 16,       1, 256

    #-----------------------------
    # --- main part
    # ----------------------------
    save_name = os.path.split(sample["mask_path"])[-1][:-4]
    raw_image = Image.fromarray(sample["raw_image"].astype('uint8')).convert('RGB')

    # ------ DO NOT CHANGE THE CODE BELOW
    for bid in range(NUMBER_BATCHES):  # parallel sampling
        # ------ guiding synthesis
        guide_start_time = time.time()

        fake_z_guide = model.autoregressive_sample_fast256(z_indices_guide, c_indices_guide,
                                                            c_code_guide.shape[2], c_code_guide.shape[3],
                                                            latent_mask_guide, batch_size=NUMBER_SAMPLES,
                                                            temperature=temperature, top_k=top_k)
                                                                                                        
        print('guide synthesis %.2f seconds' % (time.time() - guide_start_time))
        
        fake_image_guide = model.decode_to_img(fake_z_guide, (fake_z_guide.shape[0], z_code_guide.shape[1], z_code_guide.shape[2], z_code_guide.shape[3]))
        for sid in range(fake_z_guide.shape[0]):
            fake_img_pil = show_output(fake_image_guide[sid:(sid + 1)])
            fake_img_pil_blended = laplacian_blend(fake_img_pil, raw_image, mask_guide, num_levels=5)
            fake_img_np = np.array(fake_img_pil_blended)
            temp_ofn = os.path.join(collect_dir, save_name + '_pred.jpg')
            write_cv2_img_jpg(fake_img_np[:, :, ::-1], temp_ofn)

        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_test_path', '-t', type=str, required=True, help='path to config file.')
    parser.add_argument('--collect_dir', '-r', type=str, default='./results', help='directory to save results.')
    args = parser.parse_args()

    test_config = OmegaConf.load(args.config_test_path)
    collect_dir = args.collect_dir

    the_seed = test_config['the_seed']
    config_path = test_config['config_path']
    ckpt_path = test_config['ckpt_path']
    NUMBER_BATCHES = test_config['NUMBER_BATCHES']
    NUMBER_SAMPLES = test_config['NUMBER_SAMPLES']
    temperature = test_config['temperature']
    top_k = test_config['top_k']

    #----- 
    torch.manual_seed(the_seed)
    torch.cuda.manual_seed(the_seed)
    torch.cuda.manual_seed_all(the_seed)  # if you are using multi-GPU.
    np.random.seed(the_seed)  # Numpy module.
    random.seed(the_seed)  # Python random module.
    torch.manual_seed(the_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    config = OmegaConf.load(config_path)
    config['model']['params']['ckpt_path'] = ckpt_path

    print(yaml.dump(OmegaConf.to_container(config)))
    model = Net2NetTransformer(**config.model.params)

    model.cuda().eval()
    torch.set_grad_enabled(False)

    assert isinstance(top_k, int)
    guide_scaler = albumentations.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_AREA)

    #-----------------------------
    # --- load data
    # ----------------------------
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    create_folder(collect_dir)

    with torch.no_grad():
        for index,sample in tqdm(enumerate(data.datasets['train'])):
            predict(sample)

        for index,sample in tqdm(enumerate(data.datasets['validation'])):
            predict(sample)

