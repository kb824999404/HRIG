import taichi as ti
import numpy as np
from tqdm import tqdm

import threading
from multiprocessing import Pool
from itertools import compress

# 多进程变量
Lock = threading.RLock()
ProcessCount = 2   #进程池大小

# 雨滴粒子模拟器数据类-可序列
class SimulatorData:
    def __init__(self,simulator,usePorjection=False):
        self.rainDropPositions = simulator.rainDropPositions.to_numpy().tolist()
        self.rainDropPositionsEnd = simulator.rainDropPositionsEnd.to_numpy().tolist()
        self.rainDropDiameters = simulator.rainDropDiameters.to_numpy()
        self.rainDropVelocitys = simulator.rainDropVelocitys.to_numpy()
        self.rainDropFalling = simulator.rainDropFalling.to_numpy()

        if usePorjection:
            self.imageDiameters = simulator.imageDiameters.to_numpy().tolist()
            self.imagePositionsStart = simulator.imagePositionsStart.to_numpy().tolist()
            self.imagePositionsEnd = simulator.imagePositionsEnd.to_numpy().tolist()
            self.viewPositions = simulator.viewPositions.to_numpy().tolist()
            self.rainDropInView = simulator.rainDropInView.to_numpy()
            self.linearZ = simulator.linearZ.to_numpy()

            self.cond = np.logical_and( self.rainDropFalling == 1, self.rainDropInView == 1 )
        else:
            self.cond = self.rainDropFalling == 1

        self.drops_count = simulator.drops_count

# 雨滴粒子模拟器
class RainDropSimulator:
    # Rain Intensity: 降雨强度，单位时间内输送到单位地表面积上的雨水体积，mm/h
    # Rain Quantity: 雨滴总数量，每平方米分配的点数量
    # Diameters range: 雨滴直径大小范围, m
    # Rain Speed: 雨滴下落速度，m/s
    # Cloud Position：云层位置，vec3
    # Cloud Size: 云层大小
    # Wind Strength: 风力大小
    # Wind Orentation: 风力方向，vec3
    # Time Speed: 动画时间速度
    # Time Offset: 降雨时间偏移量，秒
    # Time Period: 降雨周期时长，秒
    # FPS: 帧率
    # Exposure Time: 一帧曝光时间
    # Use Parallel：导出时使用多进程并行
    def __init__(self, rain_intensity: int = 1, rain_quantity:int = 100, 
                 diameters_range:list = [1e-4,1e-2],
                 cloud_position:list = [0,0,1], cloud_size:list =[1,1],
                 wind_strength:float = 0, wind_orientation:list = [0,0,0],
                 time_speed:float = 1, time_offset:list = [0,0], time_period:float = 1.0,
                 fps:int = 24,exposure_time:float = 1.0/60, useParallel=False):
        self.rain_intensity = rain_intensity
        self.rain_quantity = rain_quantity
        self.diameters_range = ti.math.vec2(diameters_range)
        self.cloud_position = ti.math.vec3(cloud_position)
        self.cloud_size = ti.math.ivec2(cloud_size)
        self.time_speed =time_speed
        self.time_offset = ti.math.vec2(time_offset)
        self.time_period = time_period
        self.fps = fps
        self.spf = 1.0 / fps        # 一帧持续多少秒
        self.exposure_time = exposure_time
        self.current_time = 0.0       # 当前时间

        # 雨滴总数量
        self.drops_count = self.rain_quantity * self.cloud_size[0] * self.cloud_size[1]
        # 雨滴当前位置
        self.rainDropPositions = ti.Vector.field(3, dtype=ti.f64, shape = self.drops_count)
        # 雨滴帧结束位置
        self.rainDropPositionsEnd = ti.Vector.field(3, dtype=ti.f64, shape = self.drops_count)
        # 雨滴初始位置
        self.rainDropInitPositions = ti.Vector.field(3, dtype=ti.f64, shape = self.drops_count)
        # 雨滴直径
        self.rainDropDiameters = ti.field(dtype=ti.f64, shape = self.drops_count)
        # 雨滴下落速度
        self.rainDropVelocitys = ti.field(dtype=ti.f64, shape = self.drops_count)
        # 雨滴是否在下落中
        self.rainDropFalling = ti.field(dtype=ti.i8, shape = self.drops_count)
        # 是否使用相机投影
        self.useProjection = False
        # 是否使用多进程
        self.useParallel = useParallel

        # 计算风力
        if wind_strength > 0  and np.linalg.norm(wind_orientation) > 0:
            wind_accelerate = (wind_strength/ np.linalg.norm(wind_orientation)) * np.array(wind_orientation)
        else:
            wind_accelerate = np.zeros(3)
        self.wind_accelerate = ti.math.vec3(wind_accelerate)

        # 初始化雨滴位置，直径和速度
        self.initRainDrops()

    # 设置相机相关参数
    # Resulotion: 图像分辨率
    # View Matrix: 视图矩阵
    # Projection Matrix：投影矩阵
    # Clip Range: 裁剪距离范围
    def setCameraSettings(self, view_matrix, projection_matrix, fov,
                          resolution = [1920,1080],
                          clip_range = [0.1,100]):
        self.useProjection = True
        self.view_matrix = ti.math.mat4(view_matrix)
        self.projection_matrix = ti.math.mat4(projection_matrix)
        self.fov = fov
        self.resolution = ti.math.vec2(resolution)
        self.clip_range = ti.math.vec2(clip_range)

        # 雨滴是否在视野内
        self.rainDropInView = ti.field(dtype=ti.i8, shape = self.drops_count)
        # 雨滴帧开始位置(图像平面)
        self.imagePositionsStart = ti.Vector.field(3, dtype=ti.f64, shape = self.drops_count)
        # 雨滴帧结束位置(图像平面)
        self.imagePositionsEnd = ti.Vector.field(3, dtype=ti.f64, shape = self.drops_count)
        # 雨滴直径(图像平面)
        self.imageDiameters = ti.field(dtype=ti.f64, shape = self.drops_count)
        # 雨滴深度值(裁剪空间，0~1，线性关系)
        self.linearZ = ti.field(dtype=ti.f64, shape = self.drops_count)
        # 雨滴相机空间坐标
        self.viewPositions = ti.Vector.field(3, dtype=ti.f64, shape = self.drops_count)

    # 初始化雨滴
    def initRainDrops(self):
        # 初始化雨滴位置
        init_positions( self.rain_intensity, self.rain_quantity,
                        self.cloud_size, self.cloud_position,
                        self.rainDropPositions, self.rainDropInitPositions )
        # 初始化雨滴直径和速度
        init_diameters_velocitys(self.rain_intensity, self.diameters_range, 
                                 self.rainDropDiameters, self.rainDropVelocitys)

    # 更新雨滴位置
    def updatePositions(self, time):
        self.current_time = time
        update_positions(time, self.time_speed, self.time_offset, self.time_period, 
                         self.cloud_position,
                         self.wind_accelerate,
                         self.rainDropPositions, self.rainDropInitPositions,
                         self.rainDropVelocitys, self.rainDropFalling)
        update_positions_end(self.rainDropPositions,self.rainDropVelocitys,
                            self.exposure_time,self.wind_accelerate,self.rainDropPositionsEnd)
        if self.useProjection:
            update_projection(self.rainDropPositions, self.rainDropPositionsEnd,
                              self.imagePositionsStart, self.imagePositionsEnd,self.linearZ,self.viewPositions,
                              self.rainDropDiameters, self.imageDiameters,
                              self.rainDropFalling, self.rainDropInView,
                              self.view_matrix,self.projection_matrix,
                              self.fov,self.resolution,self.clip_range)

    # 导出当前时刻的所有雨滴
    def exportCurrent(self):
        streaks = []

        rainDropPositions = self.rainDropPositions.to_numpy().tolist()
        rainDropPositionsEnd = self.rainDropPositionsEnd.to_numpy().tolist()
        rainDropDiameters = self.rainDropDiameters.to_numpy()
        rainDropVelocitys = self.rainDropVelocitys.to_numpy()
        rainDropFalling = self.rainDropFalling.to_numpy()

        if self.useProjection:
            imageDiameters = self.imageDiameters.to_numpy().tolist()
            imagePositionsStart = self.imagePositionsStart.to_numpy().tolist()
            imagePositionsEnd = self.imagePositionsEnd.to_numpy().tolist()
            viewPositions = self.viewPositions.to_numpy().tolist()
            rainDropInView = self.rainDropInView.to_numpy()
            linearZ = self.linearZ.to_numpy()

            cond = np.logical_and( rainDropFalling == 1, rainDropInView == 1 )
            for index in tqdm(range(self.drops_count)):
                if cond[index]:
                    streak = {
                        "pid": index,
                        "diameter": rainDropDiameters[index],
                        "velocity": rainDropVelocitys[index],
                        "position_start":  rainDropPositions[index],
                        "position_end": rainDropPositionsEnd[index],
                        "image_position_start": imagePositionsStart[index],
                        "image_position_end": imagePositionsEnd[index],
                        "image_diameter": imageDiameters[index],
                        "linear_z": linearZ[index],
                        "view_position": viewPositions[index],
                    }
                    streaks.append(streak)
        else:
            cond = rainDropFalling == 1
            for index in tqdm(range(self.drops_count)):
                if cond[index]:
                    streak = {
                        "pid": index,
                        "diameter": rainDropDiameters[index],
                        "velocity": rainDropVelocitys[index],
                        "position_start":  rainDropPositions[index],
                        "position_end": rainDropPositionsEnd[index],
                    }
                    streaks.append(streak)
        frame = {
            "start_time": self.current_time,
            "exposure_time": self.exposure_time,
            "streak_count": len(streaks),
            "streaks": streaks
        }
        return frame

    # 导出当前时刻的所有雨滴-多进程
    def exportCurrentMulti(self):

        data = SimulatorData(self,self.useProjection)
        pool = Pool(processes=ProcessCount)
        print("Create Processes")
        rainDrops = [ (data,index) for index in tqdm(range(self.drops_count))  ]
        print("Start Processes")

        if self.useProjection:
            result = pool.map(export_process_projection,rainDrops)
        else:
            result = pool.map(export_process,rainDrops)
        pool.close()
        pool.join()

        # 筛选正在下落和在视野中的雨滴
        streaks = list(compress(result,data.cond))

        frame = {
            "start_time": self.current_time,
            "exposure_time": self.exposure_time,
            "streak_count": len(streaks),
            "streaks": streaks
        }
        return frame

    # 导出某一秒的所有雨滴
    def exportAtSecond(self, second):
        self.updatePositions(second)
        if self.useParallel:
            return self.exportCurrentMulti()
        else:
            return self.exportCurrent()
        

    # 导出某一帧的所有雨滴
    def exportAtFrame(self, frame):
        self.updatePositions(frame * self.spf)
        if self.useParallel:
            return self.exportCurrentMulti()
        else:
            return self.exportCurrent()

# 导出雨滴的进程
def export_process_projection(item):
    data,index = item
    # if index % 1000 == 0:
        # print("Index: {}/{}".format(index,data.drops_count))
    if data.cond[index]:
        streak = {
            "pid": index,
            "diameter": data.rainDropDiameters[index],
            "velocity": data.rainDropVelocitys[index],
            "position_start":  data.rainDropPositions[index],
            "position_end": data.rainDropPositionsEnd[index],
            "image_position_start": data.imagePositionsStart[index],
            "image_position_end": data.imagePositionsEnd[index],
            "image_diameter": data.imageDiameters[index],
            "linear_z": data.linearZ[index],
            "view_position": data.viewPositions[index],
        }
        return streak
    else:
        return None

# 导出雨滴的进程
def export_process(item):
    data,index = item
    # if index % 1000 == 0:
        # print("Index: {}/{}".format(index,data.drops_count))
    if data.cond[index]:
        streak = {
            "pid": index,
            "diameter": data.rainDropDiameters[index],
            "velocity": data.rainDropVelocitys[index],
            "position_start":  data.rainDropPositions[index],
            "position_end": data.rainDropPositionsEnd[index],
        }
        return streak
    else:
        return None


# 初始化雨滴位置
@ti.kernel
def init_positions( rain_intensity: ti.i32, rain_quantity: ti.i32,
                    cloud_size: ti.math.ivec2, cloud_position: ti.math.vec3,
                    positions: ti.template(), initPositions: ti.template()):
    print("Initing positions for {} raindrops...".format(positions.shape[0]))
    rain_density = calc_density(rain_intensity)
    # print("Density:",rain_density)
    halfSize = ti.math.vec2(cloud_size[0]/2,cloud_size[1]/2)
    for row,col in ti.ndrange(cloud_size[0],cloud_size[1]):
        cloud_index = row * cloud_size[1] + col
        offset = ti.math.vec2(row,col) - halfSize
        for drop_index in ti.ndrange(rain_quantity):
            index = cloud_index * rain_quantity + drop_index
            initPositions[index] = [ offset[0] + ti.random(ti.f64), offset[1] + ti.random(ti.f64),
                                            ( drop_index + float(cloud_index) / (cloud_size[0]*cloud_size[1]) ) / rain_density ]
            positions[index] = cloud_position + initPositions[index]
            if positions[index][2] > cloud_position[2]:
                positions[index][2] = cloud_position[2]
            # print("row:{}\tcol:{}\tcloud_index:{}\tdrop_index:{}\tindex:{}\tPos:{}".format(row,col,cloud_index,drop_index,index,rainDropPositions[index]))
    print("Init positions for {} raindrops done.".format(positions.shape[0]))


# 根据降雨强度计算雨滴密度，即每立方米包含的雨滴数量
@ti.func
def calc_density(I:ti.i32) -> ti.i32:
    return ti.round(172 * ti.pow(I,0.22)) 

# 初始化雨滴直径和速度
@ti.kernel
def init_diameters_velocitys( rain_intensity:ti.i32, diameters_range:ti.math.vec2,
                            diameters: ti.template(), velocitys: ti.template()):
    print("Initing diameters and velocitys for {} raindrops...".format(diameters.shape[0]))

    # 计算RSD的系数
    # N(D) = A*e^{-βD}
    beta = 4100 * ti.pow(rain_intensity,-0.21)
    e_min = ti.exp(-beta*diameters_range[0])
    e_diff = e_min - ti.exp(-beta*diameters_range[1])

    for index in diameters:
        diameters[index] = calc_diameter(ti.random(ti.f64),beta,e_min,e_diff)
        velocitys[index] = calc_velocity(D=diameters[index])
        # print("Index:{}\tD:{}\tv:{}".format(index,diameters[index],velocitys[index]))
    print("Init diameters and velocitys for {} raindrops done.".format(diameters.shape[0]))


# 根据累计分布函数的逆函数计算直径
# u: 累计分布函数值，均匀随机采样得到，[0,1]
# 返回值：雨滴直径
@ti.func
def calc_diameter(u:ti.f64, beta:ti.f64, e_min:ti.f64, e_diff:ti.f64) -> ti.f64:
    return -ti.math.log( e_min - u*e_diff )/beta

# 根据雨滴直径计算下落速度
@ti.func
def calc_velocity(D:ti.f64) -> ti.f64:
    return 130 * ti.sqrt(D)


# 根据时间更新雨滴位置
@ti.kernel
def update_positions(time: ti.f64, time_speed: ti.f64, time_offset: ti.math.vec2, time_period: ti.f64, 
                     cloud_position: ti.math.vec3,
                     wind_accelerate: ti.math.vec3,
                     positions: ti.template(), initPositions: ti.template(), 
                     velocitys: ti.template(), falling: ti.template()):

    for index in positions:
        time_in_period = ti.math.mod(time + time_offset[0], time_period)
        offsetZ = initPositions[index][2] - time_speed * ( time_in_period + time_offset[1] )

        # 设置雨滴在雨层中的位置
        positions[index] = cloud_position + initPositions[index]
        positions[index][2] = cloud_position[2]
        # 未轮到该雨滴下落，留在云层平面中
        falling[index] = ti.i8(0)
        # 轮到该雨滴下落，根据速度下落
        if offsetZ < 0:
            positions[index][2] += offsetZ * velocitys[index]
            falling[index] = ti.i8(1)
        # 如果到达地面，不再下落
        if positions[index][2] < 0:
            positions[index][2] = 0
            falling[index] = ti.i8(0)

        # 添加风的影响
        positions[index] += (cloud_position[2] - positions[index][2]) * wind_accelerate

# 更新雨滴帧结束位置
@ti.kernel
def update_positions_end(   positions: ti.template(), velocitys: ti.template(), 
                            exposure_time: ti.f64, wind_accelerate: ti.math.vec3,
                            positionsEnd: ti.template()):
    for index in positions:
        positionsEnd[index] = positions[index] + exposure_time * ti.math.vec3(
            wind_accelerate[0], wind_accelerate[1], -velocitys[index]
        )


# 将雨滴投影至图像平面
@ti.kernel
def update_projection(  positions: ti.template(), positionsEnd: ti.template(),
                        imagePositionsStart: ti.template(), imagePositionsEnd: ti.template(), linearZ: ti.template(), viewPosition: ti.template(),
                        diameters: ti.template(), imageDiameters: ti.template(),
                        falling: ti.template(), inView: ti.template(),
                        view_matrix: ti.template(), projection_matrix: ti.template(),
                        fov: ti.f64, resolution: ti.math.vec2, clip_range: ti.math.vec2):
    for index in positions:
        if falling[index] == 1:
            pos_start = ti.math.vec4(positions[index],1.0)
            pos_end = ti.math.vec4(positionsEnd[index],1.0)
            view_start = view_matrix @ pos_start
            view_end = view_matrix @ pos_end
            image_start = projection_matrix @ view_start    # Clip Space
            image_end = projection_matrix @ view_end        # Clip Space
            image_start /= image_start[3]                   # NDC [-1,1]
            image_end /= image_end[3]                       # NDC [-1,1]
            # 判断是否在视野内
            if view_start[2] < -clip_range[0] and view_start[2] > -clip_range[1] and \
                view_end[2] < -clip_range[0] and view_end[2] > -clip_range[1] and \
                image_start[0] >= -1.0 and image_start[0] <= 1.0 and image_start[1] >= -1.0 and image_start[1] <= 1.0 and \
                image_end[0] >= -1.0 and image_end[0] <= 1.0 and image_end[1] >= -1.0 and image_end[1] <= 1.0:
                inView[index] = ti.i8(1)
                imagePositionsStart[index] = ti.math.vec3( (image_start[:2] + 1.0) / 2.0 * resolution, image_start[2])
                imagePositionsEnd[index] = ti.math.vec3( (image_end[:2] + 1.0) / 2.0 * resolution, image_end[2])
                viewZ_pos = -(view_start[2]+view_end[2])/2
                linearZ[index] = ti.math.clamp( viewZ_pos / clip_range[1] ,xmin=0.0,xmax=1.0)
                viewPosition[index] = (view_start.xyz+view_end.xyz)/2
                # 计算球体投影到像素空间的直径大小：https://www.cnblogs.com/charlee44/p/16684840.html
                imageDiameters[index] = ti.abs( diameters[index] * resolution[1] / ( (image_start[2]+image_end[2])*ti.math.tan(fov/2) ) )
