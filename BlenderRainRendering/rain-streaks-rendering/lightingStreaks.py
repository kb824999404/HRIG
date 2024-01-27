import bpy
import mathutils

import numpy as np
import os
from tqdm import tqdm

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
    parser = ArgumentParserForBlender(description='Lighting Rain Streaks')

    parser.add_argument('-O', '--output',
                    type=str,
                    default='../data/output',
                    help='The output directory')
    parser.add_argument('--scene',
                    type=str,
                    default='lane',
                    help='The scene name')
    parser.add_argument('--sequence',
                    type=str,
                    default='front',
                    help='The sequence name')
    parser.add_argument('-i', '--intensity',
                        help='Rain Intensity',
                        type=str,
                        default='10')
    parser.add_argument('--names',
                    type=str,
                    default=None,
                    help='The output particles names; if none, use all names in the intensity directory')
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

    args = parser.parse_args()

    if args.config:
        assert os.path.exists(args.config), ("The config file is missing.", args.config)
        with open(args.config,"r") as f:
            cfg = yaml.safe_load(f)['config']
        for key in cfg:
            args.__dict__[key] = cfg[key]

    return args


# 创建雨纹面片
def createQuad(name,buffer_mask,buffer_pos, hide=False):
    print("Creating Rain Quads...")
    vertexs = []
    indices = []
    height, width = buffer_mask.shape
    for y in tqdm(range(height)):
        for x in range(width):
            if buffer_mask[y,x] == True:
                start_index = len(vertexs)
                face = []
                for i in range(4):
                    vertexs.append(tuple(buffer_pos[y,x][i]))
                    face.append(start_index+i)
                indices.append(face)


    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertexs, [], indices)
    obj = bpy.data.objects.new(name,mesh)
    obj.location = (0,0,0)
    scene.collection.objects.link(obj)
    if hide == True:
        obj.hide_viewport = True
        obj.hide_render = True

    return obj,mesh

# 删除物体和网格
def removeObj(obj, mesh):
    scene.collection.objects.unlink(obj)
    bpy.data.objects.remove(obj)
    bpy.data.meshes.remove(mesh)

def renderStreaks(path):
    print("Rendering rain streaks in:",path)
    buffer_path = os.path.join(path,"buffers")
    frame_count = int(np.ceil((frame_end - frame_start + 1) / frame_step))
    if args.frame_end:
        end_count = min(frame_count,args.frame_end)
    else:
        end_count = frame_count
    for index,frame in tqdm(enumerate(range(frame_start,frame_end+1,frame_step))):
        if index < args.frame_start:
            continue
        if index >= end_count:
            break
        # 设置当前帧
        scene.frame_set(frame)
        # 加载雨纹数据
        data = np.load(os.path.join(buffer_path,"frame_{:04d}.npz".format(frame)))
        buffer_mask, buffer_pos = data['buffer_mask'], data['buffer_pos_rec']
        # 在场景中创建雨纹面片
        obj, mesh = createQuad("rainStreaks_{:4d}".format(frame),buffer_mask,buffer_pos)

        # 输出
        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = os.path.join(path,'rain_lighting','frame_{:04d}.png'.format(frame))
        
        # 渲染图片并保存为文件
        bpy.ops.render.render( write_still=True )

        removeObj(obj,mesh)


if __name__=="__main__":
    args = get_args()

    intensity_path = os.path.join(args.output,args.scene,args.sequence,args.intensity+'mm')
    assert os.path.exists(intensity_path), ("intensity output path is missing.", intensity_path)


    if args.names:
        names = [ name for name in args.names.split(',') ]
    else:
        names = os.listdir(intensity_path)
        names = [  name for name in names if os.path.isdir(os.path.join(intensity_path,name)) ]

    # 获取场景变量
    scene = bpy.context.scene
    frame_start = scene.frame_start
    frame_end = scene.frame_end
    frame_step = scene.frame_step

    ################### 渲染相关设置 ###################
    # 色彩管理
    scene.display_settings.display_device = 'None'              # 显示设备：无
    scene.view_settings.look = 'None'                           # 胶片效果：无
    scene.sequencer_colorspace_settings.name = 'Linear'         # 序列编辑器：线性

    for name in names:
        path = os.path.join(intensity_path,name)
        if os.path.exists(path):
            renderStreaks(path)

