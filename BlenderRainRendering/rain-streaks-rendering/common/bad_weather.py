import os
import sys
import traceback
from enum import Enum

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

from common import my_utils

plt.ion()

try:
    import cPickle as pickle
except Exception:
    import pickle


'''
###########################################
#       IMPORTANT                         #
###########################################                                        

- The positive direction of the Z-Axis is being reversed in our code as compared to the one in simulation file
- The positive direction of the Y-Axis is being reversed in our code as compared to the KITTI dataset

## Conventions (comments about papers)
# vr - Vision and Rain (Garg & Nayar 2007)
# pr - Photorealistic Rendering of Rain Streaks (Garg & Nayar 2006)
'''

cache = {}


class DropType(Enum):
    Big = 0
    Medium = 1
    Small = 2

# 雨纹类
class Streak:
    def __init__(self, ):
        self.pid = None
        self.world_position_start = None    # 帧起始雨滴位置(世界坐标系)
        self.world_position_end = None      # 帧结束雨滴位置(世界坐标系)
        self.world_diameter_start = None    # 帧起始雨滴直径(世界坐标系)
        self.world_diameter_end = None      # 帧结束雨滴直径(世界坐标系)
        self.image_position_start = None    # 帧起始雨滴位置(图像坐标系)
        self.image_position_end = None      # 帧结束雨滴位置(图像坐标系)
        self.linear_z = None                # 雨滴深度值(裁剪空间，0~1，线性关系)
        self.view_position = None           # 相机空间坐标
        self.image_diameter_start = None    # 帧起始雨滴直径(图像坐标系)
        self.image_diameter_end = None      # 帧结束雨滴直径(图像坐标系)
        self.ratio = None                   # 雨滴直径与雨纹长度之比
        self.max_width = None               # 开始和结束时间中最大的直径
        self.length = None                  # 雨纹长度整数值
        self.drop_type = None               # 雨滴类型：小，中，大 三种

    def __repr__(self):
        return str(self.__dict__).replace(',', '\n')

# 帧类
class Frame:
    def __init__(self, ):
        self.id = None
        self.starting_time = None       # 开始时间
        self.exposure_time = None       # 曝光时间
        self.streaks_count = None       # 雨纹数量
        self.streaks = None

    def __repr__(self):
        return str(self.__dict__).replace(',', '\n')

# 雨纹数据管理类
class DBManager:
    # def __init__(self, streaks_path=None, streaks_path_xml=None, norm_coeff_path=None):
    # def __init__(self, streaks_path_xml=None, streaks_db=None, streaks_size=None):
    def __init__(self, streaks_path_pkl=None, streaks_db=None, streaks_size=None):
        '''
        Function to initialize the class.
        :param streaks_path: Path to the light texture database.
        :param streaks_path_xml: Path to the output of the simulator.
        :param norm_coeff_path: TODO:: FILL
        '''
        self.streaks_path = os.path.join(streaks_db,'env_light_database',streaks_size)
        self.streaks_path_point = os.path.join(streaks_db,'point_light_database',streaks_size)
        self.norm_coeff_path = os.path.join(streaks_db, 'env_light_database', 'txt', 'normalized_env_max.txt')
        self.norm_coeff_path_point = os.path.join(streaks_db, 'point_light_database', 'txt')
        # self.streaks_path_xml = streaks_path_xml
        self.streaks_path_pkl = streaks_path_pkl
        self.streaks_light = np.array([])
        self.streaks_light_point = {}
        self.streaks_simulator = {}
        self.ratio = np.array([])

    def __repr__(self):
        return "DataAcquisition()"

    def __str__(self):
        return 'DataAcquisition'.format()

    @staticmethod
    def classify_drop(w):
        if w >= 4:
            return DropType(0)
        if w > 1:
            return DropType(1)

        return DropType(2)
    
    # 加载环境光雨纹数据
    def load_streak_database(self):
        '''
        Function to load and store the texture maps.
        Streaks are stored in a list.
        '''

        if not os.path.exists(self.streaks_path):
            print("No existing path for streak database (", self.streaks_path, ")")
            exit(-1)

        tmp = [] 
        norm_coeff_path = self.norm_coeff_path
        norm_coeffs = {}

        with open(norm_coeff_path, 'r') as file:
            lines = file.readlines()
        
        # 读取亮度系数
        for line in lines:
            if line[:2] == 'cv':
                coeff = int(line[2:])
                continue
            norm_coeffs.update({coeff: [float(v) for v in line.split('\n')[0].split(' ')[:-1]]})

        # 读取雨纹图像，两个参数：视角方向coeff和振荡类型osc
        for file_name in my_utils.os_listdir(self.streaks_path):
            name = os.path.splitext(file_name)[0]
            coeff, osc = name.split('_')
            if len(coeff) == 3:
                coeff = int(coeff[-1:])
            else:
                coeff = int(coeff[-2:])
            osc = int(osc[-1:])
            # 16位图像
            drop_image = cv2.imread(os.path.join(self.streaks_path, file_name), cv2.IMREAD_ANYDEPTH)
            drop_image = cv2.cvtColor(drop_image, cv2.COLOR_GRAY2BGR)
            # 16位图像转8位图像
            drop_image_norm = ((255.0 * norm_coeffs[coeff][osc] * drop_image) / 65535.0).astype(np.uint8)
            tmp.append(drop_image_norm)
            # 计算图像宽高比
            self.ratio = np.append(self.ratio, tmp[-1].shape[1] / tmp[-1].shape[0])

        # 所有不重复的雨纹图像宽高比
        self.ratio = np.unique(self.ratio)
        # 所有雨纹图像
        self.streaks_light = tmp

    # 加载点光源雨纹数据
    def load_streak_database_point(self):
        '''
        Function to load and store the texture maps.
        Streaks are stored in a list.
        '''

        if not os.path.exists(self.streaks_path_point):
            print("No existing path for streak database (", self.streaks_path_point, ")")
            exit(-1)

        norm_coeff_path_point = self.norm_coeff_path_point
        norm_coeffs = {}

        
        coeffs = [ '00', '20', '40', '60', '80' ]
        for coeff in coeffs:
            path = os.path.join(norm_coeff_path_point,'dcam{}_point_max.txt'.format(coeff))
            with open(path, 'r') as file:
                lines = file.readlines()
                light_vertical = None
                light_horizontal = None
                norm_coeff = {}
                light_coeff = {}
                for line in lines[1:]:
                    if line[:1] == 'v':
                        light_vertical = int(line.split('\n')[0][1:])
                        norm_coeff[light_vertical] = {}
                        light_coeff[light_vertical] = {}
                        continue
                    elif line[:1] == 'h':
                        items = line.split('\n')[0].split(' ')
                        light_horizontal = int(items[0][1:])
                        norm_coeff[light_vertical].update({light_horizontal: [float(v) for v in items[1:-1] ]})
                        light_coeff[light_vertical][light_horizontal] = []
                norm_coeffs[int(coeff)] = norm_coeff
                self.streaks_light_point[int(coeff)] = light_coeff

        # 视角方向coeff
        for coeff in coeffs:
            path = os.path.join(self.streaks_path_point,'dcam{}'.format(coeff))
            for file_name in my_utils.os_listdir(path):
                name = os.path.splitext(file_name)[0]
                # 振荡类型osc
                _, vertical, horizontal, osc = name.split('_')
                coeff = int(coeff)
                vertical = int(vertical[1:])
                horizontal = int(horizontal[1:])
                osc = int(osc[-1:])
                # 16位图像
                drop_image = cv2.imread(os.path.join(path, file_name), cv2.IMREAD_ANYDEPTH)
                drop_image = cv2.cvtColor(drop_image, cv2.COLOR_GRAY2BGR)
                # 16位图像转8位图像
                drop_image_norm = ((255.0 * norm_coeffs[coeff][vertical][horizontal][osc] * drop_image) / 65535.0).astype(np.uint8)
                self.streaks_light_point[coeff][vertical][horizontal].append(drop_image_norm)


    # 加载降雨粒子模拟文件，存储对应的雨纹
    def load_streaks_from_pkl(self, dataset, settings, image_shape_WH, use_pickle=True, verbose=True):
        if not os.path.exists(self.streaks_path_pkl):
            my_utils.print_error("No existing path for PKL file (" + self.streaks_path_pkl + ")")
            exit(-1)

        try:
            print("Reading PKL file {} ...".format(self.streaks_path_pkl))
            with open(self.streaks_path_pkl, "rb") as tf:
                simulation = pickle.load(tf)
        except Exception as e:
            raise Exception("Reading PKL file {} crashed, which is likely due to corrupted particles simulation files. If so, delete this simulation folder manually and re-run to allow generation of new simulation.".format(self.streaks_path_pkl))

        print("Parsing PKL file {} ...".format(self.streaks_path_pkl))
        if verbose:
            # 打印进度条
            my_utils.print_progress_bar(0, len(simulation))
        try:
            for fix, frame in enumerate(simulation):
                f = Frame()
                f.id = frame            # 该帧的id
                f.exposure_time = simulation[frame]["exposure_time"]    # 曝光时间
                f.starting_time = simulation[frame]["start_time"]       # 开始时间
                f.streaks_count = simulation[frame]["streak_count"]    # 雨纹数量
                f.streaks = {}
                for drop in simulation[frame]["streaks"]:
                    s = Streak()
                    s.pid = int(drop["pid"])         # 该雨纹的id
                    s.world_position_start = np.array(drop["position_start"], dtype=float) # 帧起始雨滴位置(世界坐标系)
                    s.world_position_end = np.array(drop["position_end"], dtype=float)   # 帧结束雨滴位置(世界坐标系)
                    s.world_diameter_start = float(drop['diameter'])                                  # 帧起始雨滴直径(世界坐标系)
                    s.world_diameter_end = float(drop['diameter'])                                    # 帧结束雨滴直径(世界坐标系)

                    s.image_position_start = np.array(drop["image_position_start"][:-1], dtype=float) / settings["render_scale"]  # x,y   # 帧起始雨滴位置(图像坐标系)
                    s.image_position_end = np.array(drop["image_position_end"][:-1], dtype=float) / settings["render_scale"]  # x,y     # 帧结束雨滴位置(图像坐标系)
                    s.image_diameter_start = float(drop['image_diameter']) / settings["render_scale"]       # 帧起始雨滴直径(图像坐标系)
                    s.image_diameter_end = float(drop['image_diameter']) / settings["render_scale"]         # 帧结束雨滴直径(图像坐标系)

                    s.linear_z = drop['linear_z']      # 雨滴线性深度值
                    s.view_position = drop['view_position'] #相机空间坐标

                    # 交换y轴和z轴
                    # s.world_position_start[1],s.world_position_start[2] = s.world_position_start[2],s.world_position_start[1]
                    # s.world_position_end [1],s.world_position_end [2] = s.world_position_end[2],s.world_position_end[1]

                    # 图像坐标系y轴翻转，xml中的原点在左下角，实际计算原点在左上角
                    s.image_position_start[1] = image_shape_WH[1] - s.image_position_start[1]
                    s.image_position_end[1] = image_shape_WH[1] - s.image_position_end[1]
                    # 世界坐标系z轴朝相机前
                    # s.world_position_start[2] *= -1
                    # s.world_position_end[2] *= -1
                    s.view_position[2] *= -1
                    # 起点终点之差的绝对值
                    diff = abs(s.image_position_start - s.image_position_end)
                    # 取开始和结束时间中最大的直径
                    s.max_width = int(max(s.image_diameter_start, s.image_diameter_end))

                    dir1 = np.array([0, -1])
                    # 用雨纹长度进行归一化，得到雨纹方向
                    dir2 = diff / np.linalg.norm(diff)
                    dir2[1] = -dir2[1]
                    # theta表示关于x轴的角度
                    cos_theta = np.dot(dir1, dir2)
                    # 雨纹实际长度，和直接用起点终点计算有什么区别？
                    actual_length = diff[1] / cos_theta
                    # 雨滴直径与雨纹长度之比
                    s.ratio = s.max_width / actual_length
                    # 将起点终点确定到像素
                    s.image_position_end = s.image_position_end.round().astype(int)
                    s.image_position_start = s.image_position_start.round().astype(int)
                    # 雨纹长度整数值
                    s.length = np.ceil(np.linalg.norm(s.image_position_start - s.image_position_end)).astype(int)
                    # 根据雨滴直径确定雨滴类型：小，中，大 三种
                    s.drop_type = self.classify_drop(s.max_width)
                    # 剔除直径和长度小于一个像素的雨纹
                    if s.max_width >= 1 and s.length >= 1:
                        f.streaks.update({s.pid: s})
                self.streaks_simulator.update({f.id: f})
                if verbose:
                    my_utils.print_progress_bar(fix + 1, len(simulation))
        except Exception as e:
            ex_type, ex, tb = sys.exc_info()
            my_utils.print_error('Error while parsing PKL file.\n\tFile: ' + self.streaks_path_pkl)
            traceback.print_tb(tb)
            exit(-1)

    def take_drop_texture(self, drop):
        if drop.ratio < self.ratio[0]:
            drop = self.streaks_light[np.random.randint(0, 10)] / 255.0
            return drop
        if drop.ratio < self.ratio[1]:
            drop = self.streaks_light[np.random.randint(10, 20)] / 255.0
            return drop
        if drop.ratio < self.ratio[2]:
            drop = self.streaks_light[np.random.randint(20, 30)] / 255.0
            return drop
        if drop.ratio < self.ratio[3]:
            drop = self.streaks_light[np.random.randint(30, 40)] / 255.0
            return drop
        else:
            drop = self.streaks_light[np.random.randint(40, 50)] / 255.0
            return drop
        
    def take_drop_texture_point(self, drop):
        coeff_index = len(self.ratio)-1
        # 根据宽长比选取第一个参数
        for index,ratio in enumerate(self.ratio):
            if drop.ratio < ratio:
                coeff_index = index
                break
        coeff = list(self.streaks_light_point.keys())[coeff_index]
        # 随机选择其他参数
        vertical = random.sample(self.streaks_light_point[coeff].keys(),1)[0]
        horizontal = random.sample(self.streaks_light_point[coeff][vertical].keys(),1)[0]
        osc =  random.randint(0,9)
        drop = self.streaks_light_point[coeff][vertical][horizontal][osc] / 255.0
        return drop

    @staticmethod
    def normalize(v):
        return v / np.linalg.norm(v)
