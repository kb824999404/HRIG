import argparse
import os
import sys
import warnings
import yaml

import numpy as np
from natsort import natsorted
import taichi as ti
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import taichi as ti
import taichi.math as tm

from common import my_utils

warnings.filterwarnings("ignore")

def resolve_paths(params):
    # List sequences path (relative to dataset folder)
    # Let's just consider any subfolder is a sequence
    params.sequences = [x for x in os.listdir(params.images_root) if os.path.isdir(os.path.join(params.images_root, x))]
    assert (len(params.sequences) > 0), "There are no valid sequences folder in the dataset root"

    # Set source image directory
    params.images = {s: os.path.join(params.dataset_root, s, 'background') for s in params.sequences}


    # Set depth directory
    params.depth = {s: os.path.join(params.dataset_root, s, 'depth') for s in params.sequences}

    # Set lights directory
    params.scene = {s: os.path.join(params.dataset_root, s, 'scene_info.json') for s in params.sequences}

    # Set attenuation directory
    params.attenuation = {s: os.path.join(params.attenuation_root, params.dataset, s) for s in params.sequences}

    return params


def check_arg(args):
    parser = argparse.ArgumentParser(description='Rain renderer composite')

    
    # 数据集名称
    parser.add_argument('--dataset',
                        help='Enter dataset name. Dataset data must be located in: DATASET_ROOT/DATASET',
                        type=str, required=True)

    # changed according to mode
    # RGB数据根目录
    parser.add_argument('-k', '--dataset_root',
                        help='Path to database root',
                        default=os.path.join('data', 'source'),
                        required=False)
    # RGB数据根目录
    parser.add_argument('-a', '--attenuation_root',
                        help='Path to attenuation image root',
                        default=os.path.join('data', 'attenuation'),
                        required=False)
    
    # 数据集序列名称
    parser.add_argument('-s', '--sequences',
                        help='List of sequences comma separated (e.g. for KITTI: data_object/training,data_object/testing).',
                        default='',
                        required=False)
    
    
    # 降雨强度
    # if not provided outputs results of every supported intensity
    parser.add_argument('-i', '--intensity',
                        help='Rain Intensities. List of fall rate comma-separated. E.g.: 1,15,25,50.',
                        type=str,
                        default='25',
                        required=False)

    # ways to not process all the frames in a sequence
    # 开始帧
    parser.add_argument('-fs', '--frame_start',
                        help='Frame start',
                        type=int,
                        default=0)
    # 结束帧
    parser.add_argument('-fe', '--frame_end',
                        help='Frame end',
                        type=int,
                        default=None)
    parser.add_argument('-fst', '--frame_step',
                        help='Frame step',
                        type=int,
                        default=1)
    # 指定帧
    parser.add_argument('-ff', '--frames',
                        type=str,
                        required=False,
                        default="")
    # 输出根目录
    parser.add_argument('--output',
                        default=os.path.join('data', 'output'),
                        help='Where to save the output',
                        required=False)
    # 配置文件
    parser.add_argument('-c', '--config',
                        type=str,
                        default=None,
                        help='The config file')    
    
    results = parser.parse_args(args)

    if results.config:
        assert os.path.exists(results.config), ("The config file is missing.", results.config)
        with open(results.config,"r") as f:
            cfg = yaml.safe_load(f)['config']
        for key in cfg:
            results.__dict__[key] = cfg[key]

    results.intensity = [int(i) for i in results.intensity.split(",")]
    if results.frames:
        results.frames = [int(i) for i in results.frames.split(",")]

    dataset_name = results.dataset
    results.dataset_root = os.path.join(results.dataset_root, dataset_name)
    assert os.path.exists(results.dataset_root), ("Dataset folder does not exist.", results.dataset_root)
    results.images_root = os.path.join(results.dataset_root)

    # 数据集序列列表
    sequences_filter = results.sequences.split(',')

    # 数据集参数解析，调用和数据集同名文件中的resolve_paths和settings
    results = resolve_paths(results)

    # 筛选输入参数指定的序列，因为读取的时候是读取数据集目录中所有的序列名
    # Filter sequences
    results.sequences = np.asarray([seq for seq in results.sequences if np.any([seq == _s for _s in sequences_filter])])

    results.output_root = os.path.join(results.output, results.dataset)

    # 查找指定序列指定强度下的目录
    # Build weathers to render
    results.particles = {}
    for seq in results.sequences:
        results.particles[seq] = [] 
        for i in results.intensity:
            intensity_path = os.path.join(results.output_root,seq,f'{i}mm')
            names = os.listdir(intensity_path)
            names = [  name for name in names if os.path.isdir(os.path.join(intensity_path,name)) ]
            # 找到{I}mm目录下的文件夹
            for name in names:
                results.particles[seq].append({
                    "intensity": i,
                    "name": name
                })

    return results

@ti.kernel
def rainy_blending(output: ti.template(), rain_layer: ti.template(), bg: ti.template(),
                   alpha: ti.template(), tau_one: ti.template(), exposure_time: ti.template(),
                   tau_zero: float):
    for y,x in bg:
        output[y,x] = ((1. - ((alpha[y,x] * tau_one[y,x]) / exposure_time[y,x])) * bg[y,x]) \
                        + rain_layer[y,x] * ( tau_one[y,x] / tau_zero)

def blendRainyImage(rain_layer,bg,buffer):
    drop_size = 1.16 * 1e-3
    tau_zero = np.sqrt(drop_size) / 50

    img_size = bg.shape[:2]
    ti_rain = ti.Vector.field(3,dtype=float,shape=img_size)
    ti_bg = ti.Vector.field(3,dtype=float,shape=img_size)
    ti_alpha = ti.field(dtype=float,shape=img_size)
    ti_tau_one = ti.field(dtype=float,shape=img_size)
    ti_exposure_time = ti.field(dtype=float,shape=img_size)
    ti_rain.from_numpy(rain_layer)
    ti_bg.from_numpy(bg)
    ti_alpha.from_numpy(buffer[...,0])
    ti_tau_one.from_numpy(buffer[...,1])
    ti_exposure_time.from_numpy(buffer[...,2])

    ti_output = ti.Vector.field(3,dtype=float,shape=img_size)

    rainy_blending(ti_output,ti_rain,ti_bg,ti_alpha,ti_tau_one,ti_exposure_time,tau_zero)

    return ti_output.to_numpy()

if __name__ == "__main__":
    print("\nBuilding internal parameters...")
    args = check_arg(sys.argv[1:])

    ti.init(arch=ti.gpu,default_ip=ti.i32,default_fp=ti.f64,device_memory_fraction=0.6)

    print("\nRunning composite...")

    # 遍历每个序列
    # case for any number of sequences and supported rain intensities
    for folder_idx, sequence in enumerate(args.sequences):
        print('\nSequence: ' + sequence)
        for sim_item in args.particles[sequence]:
            intensity, name = sim_item["intensity"], sim_item["name"]
            # out_seq_dir: data/output/DATASET/SEQUENCE
            out_seq_dir = os.path.join(args.output_root, sequence)
            # out_dir: data/output/DATASET/SEQUENCE/{}mm/{name}
            out_dir = os.path.join(out_seq_dir, f'{intensity}mm', name)
            # attenuation_dir: data/attenuation/DATASET/SEQUENCE/{}mm
            attenuation_dir = os.path.join(args.attenuation[sequence], f'{intensity}mm')

            # 对文件名按照数字大小排序
            files = natsorted(np.array([os.path.join(args.images[sequence], picture) for picture in my_utils.os_listdir(args.images[sequence])]))

            f_start, f_end, f_step = args.frame_start, args.frame_end, args.frame_step
            # 取设定的结束帧和数据集图像数量的最小值
            f_end = len(files) if f_end is None else min(f_end, len(files))
            # 获取要渲染的帧的索引列表
            if args.frames:
                # prone to go "boom", so we clip and remove 'wrong' ids
                idx = np.unique(np.clip(args.frames, 0, f_end - 1)).tolist()
            else:
                idx = list(range(f_start, f_end, f_step))  # to make it

            # 合成每一帧
            for frame in tqdm(idx):
                image_file = files[frame]           # 背景图像文件路径
                file_name = os.path.split(image_file)[-1]
                out_buffers_path = os.path.join(out_dir, 'buffers', '{}.npz'.format(file_name[:-4]))
                lighting_path = os.path.join(out_dir, "rain_lighting", '{}.png'.format(file_name[:-4]))
                attenuation_path = os.path.join(attenuation_dir, '{}.png'.format(file_name[:-4]))

                assert os.path.exists(attenuation_path), ("Attenuation Background Image does not exist.", attenuation_path)

                # bg = cv2.imread(image_file) / 255.0
                bg = cv2.imread(attenuation_path) / 255.0
                # BGR->RGB
                bg = bg[...,::-1]

                data = np.load(out_buffers_path)
                buffer_mask, buffer_color, buffer_pos = data['buffer_mask'], data['buffer_color'], data['buffer_pos_rec']
                
                buffer_lighting = cv2.imread(lighting_path, cv2.IMREAD_UNCHANGED)
                # BGR->RGB
                buffer_lighting = buffer_lighting[...,:3][...,::-1]
                if buffer_lighting is None:
                    print('Missing/Corrupted lighting data (%s)' % lighting_path)
                    continue
                if buffer_lighting.dtype == np.uint16:
                    buffer_lighting = buffer_lighting.astype(np.float32) / 65535.
                else:
                    buffer_lighting = buffer_lighting.astype(np.float32) / 255.
                rain_layer = buffer_color[...,0].copy()
                rain_layer = np.dstack([rain_layer,rain_layer,rain_layer,rain_layer])
                # 添加光照
                rain_layer[...,:3] *= buffer_lighting
                # 雨层颜色空间转换
                rain_layer[...,:3] = my_utils.convert_linear_to_srgb(rain_layer[...,:3])

                # 混合雨层和背景图像
                rainy_bg = blendRainyImage(rain_layer[...,:3],bg,buffer_color)
                # rainy_bg = rain_alpha * rain_layer[...,:3] + (1.0-rain_alpha) * bg
                # 无雨区域保持不变
                rainy_bg[buffer_mask == False] = bg[buffer_mask == False] 
                # 有雨区域黑色的像素设置为背景颜色
                rainy_bg = np.nan_to_num(rainy_bg)
                dark_count = 0
                for y in range(bg.shape[0]):
                    for x in range(bg.shape[1]):
                        if buffer_mask[y,x] == True and np.sum(rainy_bg[y,x]) == 0.0:
                            dark_count += 1
                            rainy_bg[y,x] = bg[y,x]
                # print("Dark Count:",dark_count)

                # 根据叠加雨层前后调整整体亮度
                rainy_bg_mean = np.mean(rainy_bg)
                bg_mean = np.mean(bg)
                difference_mean = rainy_bg_mean - bg_mean
                rainy_image = rainy_bg - difference_mean
                # 裁剪到[0,1]
                rainy_image = np.clip(rainy_image,0.0,1.0)

                out_rainy_path = os.path.join(out_dir, 'rainy_image', '{}.png'.format(file_name[:-4]))
                out_rain_path = os.path.join(out_dir, 'rain_layer', '{}.png'.format(file_name[:-4]))
                os.makedirs(os.path.dirname(out_rainy_path), exist_ok=True)
                os.makedirs(os.path.dirname(out_rain_path), exist_ok=True)

                plt.imsave(out_rainy_path,rainy_image)
                plt.imsave(out_rain_path,rain_layer)
