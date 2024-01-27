import bpy
import mathutils
import numpy as np

import taichi as ti
import os,importlib

import time,json
from tqdm import tqdm

import pickle

# 导入RainDropSimulator
base_dir = os.getcwd()
loader = importlib.machinery.SourceFileLoader("RainDropSimulator", os.path.join(base_dir,"rainDropSimulator.py"))
spec = importlib.util.spec_from_loader(loader.name, loader)
Simulator= importlib.util.module_from_spec(spec)
loader.exec_module(Simulator)


import argparse
import yaml
import sys

# Refer To: https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


def get_args():
    parser = ArgumentParserForBlender(description='Simulator Blender')

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
    parser.add_argument('-c', '--config',
                        type=str,
                        default=None,
                        help='The config file')   

    args = parser.parse_args()

    if args.config:
        assert os.path.exists(args.config), ("The config file is missing.", args.config)
        with open(args.config,"r") as f:
            cfg = yaml.safe_load(f)['config']
        for key in cfg:
            args.__dict__[key] = cfg[key]

    return args

# 获取场景信息
def getSceneInfo():
    context = bpy.context
    scene = context.scene
    render = scene.render
    camera = scene.camera
    print("Camera:",camera)

    fps = render.fps
    resolution_x = render.resolution_x
    resolution_y = render.resolution_y
    
    pixel_aspect_x = render.pixel_aspect_x
    pixel_aspect_y = render.pixel_aspect_y
    
    depsgraph = bpy.context.evaluated_depsgraph_get()

    fov = camera.data.angle

    projection_matrix = camera.calc_matrix_camera(depsgraph,
                                                    x=resolution_x,y=resolution_y,
                                                    scale_x=pixel_aspect_x,scale_y=pixel_aspect_y)

    view_matrix = camera.matrix_world.inverted()

    frame_current = scene.frame_current
    frame_start = scene.frame_start
    frame_end = scene.frame_end
    frame_step = scene.frame_step

    return fps,[resolution_x,resolution_y],fov,view_matrix,projection_matrix,frame_current,frame_start,frame_end,frame_step

# 创建点云网格
def createMesh(name,vertexs):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertexs, [], [])
    obj = bpy.data.objects.new(name,mesh)
    obj.location = (0,0,0)
    bpy.context.scene.collection.objects.link(obj)


# 获取一帧的所有雨滴的位置
def getFrameVertexs(frame):
    vertexs = []
    if frame["streak_count"] == 0:
        return None
    for streak in tqdm(frame["streaks"]):
        vertex = streak["position_start"]
        vertexs.append(vertex)    
    return vertexs


if __name__=="__main__":
    args = get_args()
    fps,resolution,fov,view_matrix,projection_matrix,frame_current,frame_start,frame_end,frame_step = getSceneInfo()

    ti.init(arch=ti.gpu,default_ip=ti.i32,default_fp=ti.f64,device_memory_fraction=0.8)

    # 场景雨滴粒子模拟器
    simulator = Simulator.RainDropSimulator(  rain_intensity=args.intensity, rain_quantity=args.quantity,
                                    cloud_position=args.cloud_position, cloud_size=args.cloud_size,
                                    wind_strength=args.wind_strength, wind_orientation=args.wind_orientation,
                                    time_offset=args.time_offset,time_period=args.time_period,fps=fps,useParallel=False )
    # 设置相机参数，只留下视野内的雨滴
    simulator.setCameraSettings( np.array(view_matrix),np.array(projection_matrix),fov )
    
    frame = simulator.exportAtFrame(frame_current)
    vertexs = getFrameVertexs(frame)
    if vertexs:
        createMesh("RainDrops_{}".format(frame_current),vertexs)
    
    