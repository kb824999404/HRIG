import argparse
import os
import sys
import warnings
import yaml

import glob2
import numpy as np

import taichi as ti

from common.generator import Generator

np.random.seed(0)
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


def settings():
    settings = {}

    settings["depth_scale"] = 1  
    settings["render_scale"] = 1 

    # Camera intrinsic parameters
    settings["cam_hz"] = 10               # Camera Hz (aka FPS)
    settings["cam_CCD_WH"] = [1242, 375]  # Camera CDD Width and Height (pixels)
    settings["cam_CCD_pixsize"] = 4.65    # Camera CDD pixel size (micro meters)
    settings["cam_WH"] = [1242, 375]      # Camera image Width and Height (pixels)
    settings["cam_focal"] = 6             # Focal length (mm)
    settings["cam_gain"] = 20             # Camera gain
    settings["cam_f_number"] = 6.0        # F-Number
    settings["cam_focus_plane"] = 6.0     # Focus plane (meter)
    settings["cam_exposure"] = 2          # Camera exposure (ms)

    # Camera extrinsic parameters (right-handed coordinate system)
    settings["cam_pos"] = [1.5, 1.5, 0.3]     # Camera pos (meter)
    settings["cam_lookat"] = [1.5, 1.5, -1.]  # Camera look at vector (meter)
    settings["cam_up"] = [0., 1., 0.]         # Camera up vector (meter)

    return settings

def check_arg(args):
    parser = argparse.ArgumentParser(description='Rain renderer method')

    # 数据集名称
    parser.add_argument('--dataset',
                        help='Enter dataset name. Dataset data must be located in: DATASET_ROOT/DATASET',
                        type=str, required=True)

    # 数据根目录
    parser.add_argument('-k', '--dataset_root',
                        help='Path to database root',
                        default=os.path.join('data', 'source'),
                        required=False)

    # 数据集序列名称
    parser.add_argument('-s', '--sequences',
                        help='List of sequences comma separated (e.g. for KITTI: data_object/training,data_object/testing).',
                        default='',
                        required=False)

    # 降雨粒子数据目录
    parser.add_argument('-r', '--particles',
                        help='Path to particles simulations',
                        default=os.path.join('data', 'particles'),
                        required=False)

    # 雨纹数据目录
    # stays the same everywhere
    parser.add_argument('-sd', '--streaks_db',
                        help='Path to rain streaks database (Garg and Nayar, 2006)',
                        default=os.path.join('3rdparty', 'rainstreakdb'),
                        required=False)
    # 配置文件
    parser.add_argument('-c', '--config',
                        type=str,
                        default=None,
                        help='The config file')    


    # 雨纹噪声
    parser.add_argument('-ns', '--noise_scale',
                        type=float,
                        default=0.0)
    parser.add_argument('-nv', '--noise_std',
                        type=float,
                        default=0.0)
    # 雨层不透明度衰减
    parser.add_argument('-oa', '--opacity_attenuation',
                        help='Opacity attenuation of the rain layer. Values must be between 0 and 1',
                        type=float,
                        default=1.0)

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

    # 如果输出目录已经存在该如何做：覆盖/跳过/重命名
    parser.add_argument('--conflict_strategy',
                        help='Strategy to use if output already exists.',
                        type=str,
                        choices=['overwrite', 'skip', 'rename_folder'],
                        default='overwrite',
                        required=False)
    # 输出根目录
    parser.add_argument('--output',
                        default=os.path.join('data', 'output'),
                        help='Where to save the output',
                        required=False)
    # 不输出日志信息
    parser.add_argument('--noverbose',
                        action='store_true')


    results = parser.parse_args(args)

    if results.config:
        assert os.path.exists(results.config), ("The config file is missing.", results.config)
        with open(results.config,"r") as f:
            cfg = yaml.safe_load(f)['config']
        for key in cfg:
            results.__dict__[key] = cfg[key]

    results.verbose = not results.noverbose

    # 雨纹数据库
    results.streaks_size = 'size32'
    assert os.path.exists(results.streaks_db), ("rainstreakdb database is missing.", results.streaks_db)

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

    # 数据集参数解析
    results = resolve_paths(results)
    results.settings = settings()

    # 筛选输入参数指定的序列，因为读取的时候是读取数据集目录中所有的序列名
    # Filter sequences
    results.sequences = np.asarray([seq for seq in results.sequences if np.any([seq == _s for _s in sequences_filter])])

    # 判断指定的序列数据是否完整，不完整则删除该序列
    # Check sequences are valid
    print("\nChecking sequences...")
    print(" {} sequences found: {}".format(len(results.sequences), [s for s in results.sequences]))
    for seq in results.sequences:
        valid = True
        if not os.path.exists(results.images[seq]):
            print(" Skip sequence '{}': images folder is missing {}".format(seq, results.images[seq]))
            valid = False
        if not os.path.exists(results.depth[seq]):
            print(" Skip sequence '{}': depth folder is missing {}".format(seq, results.depth[seq]))
            valid = False

        if not valid:
            results.sequences = results.sequences[results.sequences != seq]
            # del results.particles[seq]
            del results.images[seq]
            del results.depth[seq]

    print("Found {} valid sequence(s): {}".format(len(results.sequences), [s for s in results.sequences]))

    # 查找指定序列指定强度下的粒子文件
    particles_root = os.path.join(results.particles, results.dataset)
    results.particles = {}
    for seq in results.sequences:
        results.particles[seq] = [] 
        for i in results.intensity:
            intensity_path = os.path.join(particles_root,seq,f'{i}mm')
            names = os.listdir(intensity_path)
            names = [  name for name in names if os.path.isdir(os.path.join(intensity_path,name)) ]
            # 找到{I}mm/{name}目录下的*_frames.pkl文件
            for name in names:
                particles_file = glob2.glob( os.path.join(intensity_path,name,"*_frames.pkl") )[0]
                results.particles[seq].append({
                    "intensity": i,
                    "name": name,
                    "particles": particles_file
                })
    return results


if __name__ == "__main__":
    print("\nBuilding internal parameters...")
    args = check_arg(sys.argv[1:])

    ti.init(arch=ti.gpu,default_ip=ti.i32,default_fp=ti.f64,device_memory_fraction=0.8)

    print("\nRunning renderers...")
    generator = Generator(args)
    generator.run()
