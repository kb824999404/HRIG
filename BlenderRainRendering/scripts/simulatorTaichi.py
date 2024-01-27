import taichi as ti
import time
import os
import argparse
import yaml
from rainDropSimulator import RainDropSimulator


def get_args():
    parser = argparse.ArgumentParser(description='RainDropSimulator in Taichi')

    parser.add_argument('-c', '--config',
                        type=str,
                        default=None,
                        help='The config file')    
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

# 初始化窗口、画布、场景、相机
def initScene():
    window = ti.ui.Window("Rain Drop Simulator", (1024, 768), True, fps_limit=24)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, -20, 2)
    camera.up(0,0,1)
    camera.lookat(0,1,2)

    origin = ti.Vector.field(3,dtype=ti.f32, shape = 1)
    origin[0] = [0,0,0]

    return window,canvas,scene,camera,origin


# 绘制场景
def drawScene():
    window,canvas,scene,camera,origin = initScene()
    start_time = time.time()
    while window.running:
        # 重置时间
        if window.get_event(ti.ui.PRESS):
            if window.event.key == 'r':
                start_time = time.time()
            if window.event.key == 'z':
                break
        # 更新时间
        current_time = time.time() - start_time

        # 控制相机
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        # 设置光照
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        # 绘制原点
        scene.particles(origin,color = (1, 1, 1), radius = 0.1)

        # 绘制雨滴
        simulator.updatePositions(current_time)
        scene.particles(simulator.rainDropPositions, color = (1,0.8,0.8), radius = 0.01)
    
        canvas.scene(scene)
        window.show()



if __name__=="__main__":
    args = get_args()

    ti.init(arch=ti.gpu,default_ip=ti.i32,default_fp=ti.f64,device_memory_fraction=0.8)

    simulator = RainDropSimulator(  rain_intensity=args.intensity, rain_quantity=args.quantity,
                                    cloud_position=args.cloud_position, cloud_size=args.cloud_size,
                                    wind_strength=args.wind_strength, wind_orientation=args.wind_orientation,
                                    time_offset=args.time_offset,time_period=args.time_period,fps=24 )

    drawScene()




