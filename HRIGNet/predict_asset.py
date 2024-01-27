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
    # ------ load mask
    mask = sample["mask_0"]
    iH = mask.shape[0]
    iW = mask.shape[1]
    assert iH == iW
    assert iH in [512, 1024]  # work for 512 and 1024 square images
    mask_final = mask.copy()
    mask_final = np.clip(mask_final, 0, 1)
    mask_uint8 = np.uint8(255.0 * mask_final)  # full mask
    mask_tensor = torch.from_numpy(mask_final).to(model.device).unsqueeze(0).unsqueeze(0).to(memory_format=torch.contiguous_format)  # 1, 1, h, w, on cuda
    resized_mask_tensor = F.interpolate(mask_tensor, size=(iH // 16, iW // 16))
    latent_mask = resized_mask_tensor.squeeze().cpu().numpy()  # [0, 1]

    # ------ load natural image
    source = torch.tensor(sample["image_0"].transpose(2, 0, 1)[None]).to(dtype=torch.float32, device=model.device)  # 1, 3, h, w

    z_code, z_indices = model.encode_to_z(source, mask_tensor=mask_tensor)  # VQGAN encoding
    # print("z_code", z_code.shape, z_code.dtype)  # 1, 256, h//16, w//16
    # print("z_indices", z_indices.shape, z_indices.dtype)  # 1, h//16 * w//16

        # ------ load condition
    c_code, c_indices = model.encode_to_c(source)  # segmentation encoding
    # print("c_code", c_code.shape, c_code.dtype)  # 1, 256, h//16, w//16
    # print("c_indices", c_indices.shape, c_indices.dtype)  # 1, h//16 * w//16
    assert c_code.shape[2] * c_code.shape[3] == c_indices.shape[1]
    z_indices_shape = c_indices.shape  # 1, 32x32
    z_code_shape = c_code.shape  # 1, 256, 32, 32

    # ------ get downsampled inputs
    mask_guide = cv2.resize(mask_uint8, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_guide = mask_guide / 255.0
    mask_guide = mask_guide.astype(np.float32)
    mask_guide_tensor = torch.from_numpy(mask_guide).to(model.device).unsqueeze(0).unsqueeze(0).to(memory_format=torch.contiguous_format)  # 1, 1, 256, 256, on cuda
    resized_mask_guide_tensor = F.interpolate(mask_guide_tensor, size=(16, 16))
    latent_mask_guide = resized_mask_guide_tensor.squeeze().cpu().numpy()  # [0, 1]

    source_guide = torch.tensor(sample["image_1"].transpose(2, 0, 1)[None]).to(dtype=torch.float32, device=model.device)  # 1, 3, h, w
    z_code_guide, z_indices_guide = model.encode_to_z(source_guide, mask_tensor=mask_guide_tensor)  # VQGAN encoding

    c_code_guide, c_indices_guide = model.encode_to_c(source_guide)  # 1, 256, 16, 16,       1, 256

    #-----------------------------
    # --- main part
    # ----------------------------


    save_name = os.path.split(sample["mask_path"])[-1][:-4]

    # ------ DO NOT CHANGE THE CODE BELOW
    for bid in range(NUMBER_BATCHES):  # parallel sampling
        # ------ guiding synthesis
        guide_start_time = time.time()
        if model.is_SGA:
            fake_z_guide = model.autoregressive_sample_fast256(z_indices_guide, c_indices_guide,
                                                                c_code_guide.shape[2], c_code_guide.shape[3],
                                                                latent_mask_guide, batch_size=NUMBER_SAMPLES,
                                                                temperature=temperature, top_k=top_k)

            e_self_rand_attn, d_causal_rand_attn, d_cross_rand_attn = model.get_rough_attn_map(None, None, z_indices=fake_z_guide.reshape(-1, 256),
                                                                                                        c_indices=c_indices_guide.expand(NUMBER_SAMPLES,-1),
                                                                                                        resized_mask_tensor=resized_mask_guide_tensor.expand(NUMBER_SAMPLES,-1, -1,-1))
                                                                                                                
                                                                                                            
            print('guide synthesis %.2f seconds' % (time.time() - guide_start_time))
        # ------ high-resolution synthesis
        start_time = time.time()
        fake_z = model.autoregressive_sample_fast(z_indices, c_indices, c_code.shape[2], c_code.shape[3],
                                                latent_mask, batch_size=NUMBER_SAMPLES, temperature=temperature,
                                                top_k=top_k,
                                                e_self_rand_attn=e_self_rand_attn,
                                                d_causal_rand_attn=d_causal_rand_attn,
                                                d_cross_rand_attn=d_cross_rand_attn)

        fake_image = model.decode_to_img(fake_z, (fake_z.shape[0], z_code.shape[1], z_code.shape[2], z_code.shape[3]))
        for sid in range(fake_z.shape[0]):
            fake_img_pil = show_output(fake_image[sid:(sid + 1)])
            fake_img_pil_blended = laplacian_blend(fake_img_pil, sample["raw_image_0"], mask_final, num_levels=5)
            fake_img_np = np.array(fake_img_pil_blended)
            temp_ofn = os.path.join(collect_dir, save_name + '_pred.jpg')
            write_cv2_img_jpg(fake_img_np[:, :, ::-1], temp_ofn)

        print('target synthesis %.2f seconds' % (time.time() - start_time))
        
        


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
    guiding_ckpt_path = test_config['guiding_ckpt_path']
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
    config['model']['params']['guiding_ckpt_path'] = guiding_ckpt_path

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

