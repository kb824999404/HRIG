import numpy as np
import taichi as ti
import os, platform
import json
from tqdm import tqdm
import pickle
import time
import argparse
import yaml

from rainDropSimulator import RainDropSimulator

def get_args():
    parser = argparse.ArgumentParser(description='Export Rain Drops')

    # 输入场景参数
    parser.add_argument('-S', '--source',
                    type=str,
                    default='../data/source/lane/front/scene_info.json',
                    help='The path of the source scene info file')
                    
    # 输出目录
    parser.add_argument('-O', '--output',
                    type=str,
                    default='../data/particles',
                    help='The output directory')
    parser.add_argument('--scene',
                    type=str,
                    default='lane',
                    help='The scene name')
    parser.add_argument('--sequence',
                    type=str,
                    default='front',
                    help='The sequence name')
    parser.add_argument('--name',
                    type=str,
                    default=None,
                    help='The output particles name; if none, use the wind parameters as the name')
    parser.add_argument('-c', '--config',
                        type=str,
                        default=None,
                        help='The config file')    
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
    # 降雨参数
    parser.add_argument('-I', '--intensity',
                        type=int,
                        default=10,
                        help='The intensity of rain(mm/h)')
    parser.add_argument('-Q', '--quantity',
                    type=int,
                    default=1000,
                    help='The quantity of rain drops in unit volume')
    parser.add_argument('-cp', '--cloud_position',
                nargs='+',
                type=float,
                default=[0,0,1],
                help='The position of cloud in the scene')
    parser.add_argument('-cs', '--cloud_size',
                nargs='+',
                type=float,
                default=[1,1],
                help='The size of cloud')
    parser.add_argument('-ws', '--wind_strength',
                type=float,
                default=0,
                help='The strength of wind')
    parser.add_argument('-wo', '--wind_orientation',
                nargs='+',
                type=float,
                default=[0,0,0],
                help='The orientation of wind')
    parser.add_argument('-to', '--time_offset',
                nargs='+',
                type=float,
                default=[0,0],
                help='The time offset of rain animation(second)')
    parser.add_argument('-tp', '--time_period',
                type=float,
                default=1,
                help='The time period of rain animation(second)')
    
    args = parser.parse_args()

    if args.config:
        assert os.path.exists(args.config), ("The config file is missing.", args.config)
        with open(args.config,"r") as f:
            cfg = yaml.safe_load(f)['config']
        for key in cfg:
            args.__dict__[key] = cfg[key]

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


# 导出所有帧的雨纹为PKL
def exportAllPKL():
    frame_count = int(np.ceil((frame_end - frame_start + 1) / frame_step))
    if args.frame_end:
        end_count = min(frame_count,args.frame_end)
    else:
        end_count = frame_count
    result = {}
    for index,f in enumerate(range(frame_start,frame_end+1,frame_step)):
        if index < args.frame_start:
            continue
        if index >= end_count:
            break
        print("Exporting Frame {}/{} ...".format(index,end_count))
        view_matrix = cameras_info[str(f)]["view_matrix"]
        projection_matrix = cameras_info[str(f)]["projection_matrix"]
        fov = cameras_info[str(f)]["fov"]
        simulator.setCameraSettings( np.array(view_matrix),np.array(projection_matrix),fov )
        start_time = time.time()
        result[f] = simulator.exportAtFrame(f)
        print("Process time:",time.time()-start_time)

    resultPath_pkl = os.path.join(resultRoot,"raindrops_({}-{}-{})_frames.pkl".format(
            frame_start,frame_end,frame_step
        ))
    with open(resultPath_pkl, "wb") as tf:
        pickle.dump(result,tf)


if __name__=="__main__":
    args = get_args()
    # Check if source file exists
    assert os.path.exists(args.source), ("The source scene info file is missing.", args.source)

    # Get scene info
    with open(args.source,"r") as f:
        scene_info = json.load(f)
    fps = scene_info["render"]["fps"]
    frame_start, frame_end, frame_step = scene_info["frame_start"], scene_info["frame_end"], scene_info["frame_step"]

    # Set output path
    if args.name:
        outputName = args.name
    else:
        wo = args.wind_orientation
        outputName = "wind_{}_{}_{}_{}".format(args.wind_strength,wo[0],wo[1],wo[2])
    resultRoot = os.path.join(args.output,args.scene,args.sequence,str(args.intensity)+"mm",outputName)

    create_dir_not_exist(resultRoot)

    # Save args to file
    configs = args.__dict__
    with open(os.path.join(resultRoot,"simulator_configs.json"),"w") as f:
        json.dump(configs,f)
    

    # Init Taichi
    ti.init(arch=ti.gpu,default_ip=ti.i32,default_fp=ti.f64,device_memory_fraction=0.8)
    
    # Create Rain Drop Simulator
    simulator = RainDropSimulator(  rain_intensity=args.intensity, rain_quantity=args.quantity,
                                    cloud_position=args.cloud_position, cloud_size=args.cloud_size,
                                    wind_strength=args.wind_strength, wind_orientation=args.wind_orientation,
                                    time_offset=args.time_offset,time_period=args.time_period,fps=fps )
    
    cameras_info = scene_info["cameras"]
    exportAllPKL()
    