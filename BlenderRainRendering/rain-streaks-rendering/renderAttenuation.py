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

from common import add_attenuation, my_utils
from common.scene import SceneRenderer

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
                        default=os.path.join('data', 'attenuation'),
                        help='Where to save the output',
                        required=False)
    # 相机参数
    parser.add_argument('--f_number',
                        type=float,
                        required=False,
                        default=2,
                        help='Camera aperture size')
    parser.add_argument('--exposure',
                        type=float,
                        required=False,
                        default=1.0/60,
                        help='Camera exposure time')
    parser.add_argument('--camera_gain',
                        type=float,
                        required=False,
                        default=1,
                        help='Camera gain')
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
    results.images_root = results.dataset_root
    results.depth_root = results.dataset_root

    # 数据集序列列表
    sequences_filter = results.sequences.split(',')

    # 数据集参数解析，调用和数据集同名文件中的resolve_paths和settings
    results = resolve_paths(results)

    # 筛选输入参数指定的序列，因为读取的时候是读取数据集目录中所有的序列名
    # Filter sequences
    results.sequences = np.asarray([seq for seq in results.sequences if np.any([seq[:len(_s)] == _s for _s in sequences_filter])])

    results.output_root = os.path.join(results.output, results.dataset)

    return results


if __name__ == "__main__":
    print("\nBuilding internal parameters...")
    args = check_arg(sys.argv[1:])

    ti.init(arch=ti.gpu,default_ip=ti.i32,default_fp=ti.f64,device_memory_fraction=0.15)

    print("\nRunning composite...")

    # 遍历每个序列
    # case for any number of sequences and supported rain intensities
    for folder_idx, sequence in enumerate(args.sequences):
        print('\nSequence: ' + sequence)

        # 加载场景信息
        scene_renderer = SceneRenderer(scene_path=args.scene[sequence])
        scene_renderer.load_scene()

        # 对文件名按照数字大小排序
        files = natsorted(np.array([os.path.join(args.images[sequence], picture) for picture in my_utils.os_listdir(args.images[sequence])]))
        depth_files = natsorted(np.array([os.path.join(args.depth[sequence], depth) for depth in my_utils.os_listdir(args.depth[sequence])]))
        
        f_start, f_end, f_step = args.frame_start, args.frame_end, args.frame_step
        # 取设定的结束帧和数据集图像数量的最小值
        f_end = len(files) if f_end is None else min(f_end, len(files))
        # 获取要渲染的帧的索引列表
        if args.frames:
            # prone to go "boom", so we clip and remove 'wrong' ids
            idx = np.unique(np.clip(args.frames, 0, f_end - 1)).tolist()
        else:
            idx = list(range(f_start, f_end, f_step))  # to make it

        for intensity in args.intensity:

            # out_seq_dir: data/output/DATASET/SEQUENCE
            out_seq_dir = os.path.join(args.output_root, sequence)
            # out_dir: data/output/DATASET/SEQUENCE/{}mm/{name}
            out_dir = os.path.join(out_seq_dir, f'{intensity}mm')

            # 合成每一帧
            for f_idx in tqdm(range(len(idx))):
            # for f_idx,frame in tqdm(enumerate(idx)):
                frame = idx[f_idx]
                image_file = files[frame]           # 背景图像文件路径
                depth_file = depth_files[frame]     # 深度图像文件路径

                assert os.path.exists(image_file), "Image file {} does not exist".format(image_file)
                assert os.path.exists(depth_file), "Depth file {} does not exist".format(depth_file)

                camera = scene_renderer.cameras[f_idx]

                fog_params = {"rain_intensity": intensity, "f_number": args.f_number, "angle": np.rad2deg(camera["fov"]),
                    "exposure": args.exposure, "camera_gain": args.camera_gain}
                FOG = add_attenuation.FogRain(**fog_params)

                file_name = os.path.split(image_file)[-1]

                bg = cv2.imread(image_file) / 255.0
                # FIXME 深度值计算
                depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                if depth is None:
                    print('Missing/Corrupted depth data (%s)' % depth_file)
                    continue
                if depth.dtype == np.uint16:
                    depth = depth.astype(np.float32) / 65535.
                else:
                    depth = depth.astype(np.float32) / 255.
                depth = depth[...,0]
                depth = depth * camera['clip_end']

                # BGR->RGB
                bg = bg[...,::-1]

                rainy_bg = FOG.fog_rain_layer(bg, depth)

                out_rainy_path = os.path.join(out_dir,'{}.png'.format(file_name[:-4]))
                os.makedirs(os.path.dirname(out_rainy_path), exist_ok=True)
                plt.imsave(out_rainy_path,rainy_bg)