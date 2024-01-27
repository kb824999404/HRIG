# 雨景图像渲染主文件，调用run()运行
import os
import sys
import time

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted

from common import my_utils
from common.bad_weather import DBManager, DropType
from common.scene import SceneRenderer

plt.ion()

USE_DEPTH_CULLING = 1

class Generator:
    def __init__(self, args):
        # strategy
        self.conflict_strategy = args.conflict_strategy

        # output paths
        self.output_root = os.path.join(args.output, args.dataset)

        # dataset info
        self.dataset = args.dataset
        self.dataset_root = args.dataset_root
        self.images = args.images
        self.sequences = args.sequences
        self.depth = args.depth
        self.particles = args.particles
        self.streaks_db = args.streaks_db
        self.streaks_size = args.streaks_size
        self.settings = args.settings
        self.scene = args.scene

        # camera info
        self.exposure = args.settings["cam_exposure"]
        self.camera_gain = args.settings["cam_gain"]
        self.focal = args.settings["cam_focal"] / 1000.
        self.f_number = args.settings["cam_f_number"]
        self.focus_plane = args.settings["cam_focus_plane"]

        # aesthetic params
        self.noise_scale = args.noise_scale
        self.noise_std = args.noise_std
        self.opacity_attenuation = args.opacity_attenuation

        # generator run params
        self.frame_start = args.frame_start
        self.frame_end = args.frame_end
        self.frame_step = args.frame_step
        self.frames = args.frames
        self.verbose = args.verbose


        # initialize to None internal big frame by frame object
        self.db = None
        self.renderer = None

        # check if everything is fine
        self.check_folders()

    def check_folders(self):
        print('Output directory: {}'.format(self.output_root))

        # Verify existing folders
        existing_folders = []
        for sequence in self.sequences:
            for item in self.particles[sequence]:
                out_dir = os.path.join(self.output_root, sequence, '{}mm'.format(item['intensity']), item['name'])
                if os.path.exists(out_dir):
                    existing_folders.append(out_dir)


        if len(existing_folders) != 0 and self.conflict_strategy is None:
            print("\r\nFolders already exist: \n%s" % "\n".join([d for d in existing_folders]))
            while self.conflict_strategy not in ["overwrite", "skip", "rename_folder"]:
                self.conflict_strategy = input(
                    "\r\nWhat strategy to use (overwrite|skip|rename_folder):   ")

        assert(self.conflict_strategy in [None, "overwrite", "skip", "rename_folder"])

    # 将雨纹添加到延迟渲染的缓存中
    # drop_dict: 要渲染的雨纹
    # gbuffers: 延迟渲染的缓存，包括遮罩、深度、颜色、位置、法线、视线方向
    def compute_gbuffer(self, drop_dict, gbuffers, exposure_time, opacity_attenuation=1.0):
        buffer_mask,buffer_depth,buffer_color = gbuffers
        # 根据ratio从雨纹数据集中选择雨纹，振荡类型随机选择
        # Drop taken from database
        streak_db_drop_env = self.db.take_drop_texture(drop_dict)
        streak_db_drop_point = self.db.take_drop_texture_point(drop_dict)
        # 0.94 * F + 0.06 * E
        streak_db_drop = 0.94 * streak_db_drop_point + 0.06 * streak_db_drop_env


        image_height, image_width = buffer_mask.shape[:2]

        # Gaussian streaks do not need a perspective warping. If strak is not BIG -> Gaussian streak
        if drop_dict.drop_type == DropType.Big:
            # p1：雨纹图像四个边界点
            # p2：雨纹投射到背景图像中四个边界点
            # maxC, minC：雨纹范围坐标最大值和最小值
            # pts1, pts2, maxC, minC = self.renderer.warping_points(drop_dict, streak_db_drop, image_width, image_height)
            pts1, pts2, maxC, minC = self.scene_renderer.warping_points(drop_dict, streak_db_drop, image_width, image_height)
            shape = np.subtract(maxC, minC).astype(int)
            # 获取雨纹图像和背景图像之间的透视变换矩阵
            perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
            # 对雨纹图像进行透视变换
            drop = cv2.warpPerspective(streak_db_drop, perspective_matrix, (max(shape[0], 1), max(shape[1], 1)),
                                       flags=cv2.INTER_CUBIC)
            drop = np.clip(drop, 0, 1)
        else:
            # 用高斯噪声模拟风的影响，不需要进行透视变换，直接做旋转
            # in case of drops from database
            # Gaussian noise to simulate soft wind (in degrees)
            noise = np.random.normal(0.0, self.noise_std) * self.noise_scale

            dir1 = drop_dict.image_position_start - drop_dict.image_position_end
            n1 = np.linalg.norm(dir1)
            dir1 = dir1 / n1
            dir2 = np.array([0, -1])

            # 雨滴运动方向关于x轴的角度
            # Drop angle in degrees; add small random gaussian noise to represent localized wind
            theta = np.rad2deg(np.arccos(np.dot(dir1, dir2)))

            # Note: The noise is added to the drop coordinates AFTER the drop angle is calculated so the rotate_bound
            # function, which uses interpolation (contrarily to the drop position which are in integers),
            # would be more accurate
            nx, ny = np.cos(np.deg2rad(noise)), np.sin(np.deg2rad(noise))
            mean_x = (drop_dict.image_position_end[0] + drop_dict.image_position_start[0]) / 2
            mean_y = (drop_dict.image_position_end[1] + drop_dict.image_position_start[1]) / 2
            drop_dict.image_position_start[:] = \
                (drop_dict.image_position_start[0] - mean_x) * nx - \
                (drop_dict.image_position_start[1] - mean_y) * ny + mean_x,\
                (drop_dict.image_position_start[0] - mean_x) * ny + \
                (drop_dict.image_position_start[1] - mean_y) * nx + mean_y
            drop_dict.image_position_end[:] = \
                (drop_dict.image_position_end[0] - mean_x) * nx - \
                (drop_dict.image_position_end[1] - mean_y) * ny + mean_x,\
                (drop_dict.image_position_end[0] - mean_x) * ny + \
                (drop_dict.image_position_end[1] - mean_y) * nx + mean_y

            # 旋转雨纹
            drop = imutils.rotate_bound(streak_db_drop, theta + noise)


            drop = cv2.flip(drop, 0) if drop_dict.image_position_end[0] > image_width // 2 else drop
            height = max(abs(drop_dict.image_position_end[1] - drop_dict.image_position_start[1]), 2)
            width = max(abs(
                drop_dict.image_position_end[0] - drop_dict.image_position_start[0]), drop_dict.max_width + 2)
            # 调整雨纹分辨率
            drop = cv2.resize(drop, (width, height), interpolation=cv2.INTER_AREA)
            drop = np.clip(drop, 0, 1)
            minC = drop_dict.image_position_start

        # 雨纹为灰度图，透明度与亮度相等，保留一个通道即可
        drop = drop[..., 0]

        # 更新GBuffer
        # minC：雨纹范围坐标最小值 [minx,miny]
        # buffer中对应位置：[ minC[1]:minC[1]+drop.shape[0], minC[0]:minC[0]+drop.shape[1] ]
        if minC[0] >=0 and minC[1] >=0:
            depth = drop_dict.linear_z
            pos = drop_dict.view_position
            normal = - np.array(drop_dict.view_position)
            normal = normal / np.linalg.norm( normal )
            buffer_mask[ minC[1]:minC[1]+drop.shape[0], 
                        minC[0]:minC[0]+drop.shape[1] ] = True
            # FIXME 2023.08.20 Taichi加速循环
            for y in range(drop.shape[0]):
                for x in range(drop.shape[1]):
                    if depth < buffer_depth[minC[1]+y, minC[0]+x]:
                        # buffer_color: 3个通道，color, tau_one, exposure_time
                        buffer_color[minC[1]+y, minC[0]+x][0] = drop[y,x]

                        # compute tau_one
                        d_avg = (drop_dict.image_diameter_start + drop_dict.image_diameter_end) / 2.
                        length_opacity = opacity_attenuation * d_avg / (drop_dict.length + d_avg)
                        tau_one = exposure_time * length_opacity
                        buffer_color[minC[1]+y, minC[0]+x][1] = tau_one
                        buffer_color[minC[1]+y, minC[0]+x][2] = exposure_time
                        buffer_depth[minC[1]+y, minC[0]+x] = depth

        # FIXME 2023.08.20 处理在图像边界处的雨纹
        else:
            pass

        gbuffers = ( buffer_mask,buffer_depth,buffer_color ) 
        return gbuffers
    # 开始渲染雨景图像
    def run(self):
        process_t0 = time.time()

        folders_num = len(self.images)

        # 遍历每个序列
        # case for any number of sequences and supported rain intensities
        for folder_idx, sequence in enumerate(self.sequences):
            folder_t0 = time.time()
            print('\nSequence: ' + sequence)
            sim_num = len(self.particles[sequence])
            depth_folder = self.depth[sequence]

            # 遍历每一项粒子文件
            for sim_idx, sim_item in enumerate(self.particles[sequence]):
                intensity, name, sim_file = sim_item["intensity"], sim_item["name"], sim_item["particles"]
                # out_seq_dir: data/output/DATASET/SEQUENCE
                out_seq_dir = os.path.join(self.output_root, sequence)
                # out_dir: data/output/DATASET/SEQUENCE/{}mm/{name}
                out_dir = os.path.join(out_seq_dir, f'{intensity}mm', name)


                # 处理输出目录，不存在则创建，存在则根据指定策略解决冲突
                # Resolve output path
                path_exists = os.path.exists(out_dir)
                if path_exists:
                    if self.conflict_strategy == "skip":
                        pass
                    elif self.conflict_strategy == "overwrite":
                        pass
                    elif self.conflict_strategy == "rename_folder":
                        out_dir_, out_shift = out_dir, 0
                        while os.path.exists(out_dir_ + '_copy%05d' % out_shift):
                            out_shift += 1

                        out_dir = out_dir_ + '_copy%05d' % out_shift
                    else:
                        raise NotImplementedError

                # Create directory
                os.makedirs(out_dir, exist_ok=True)


                # 对文件名按照数字大小排序
                files = natsorted(np.array([os.path.join(self.images[sequence], picture) for picture in my_utils.os_listdir(self.images[sequence])]))
                depth_files = natsorted(np.array([os.path.join(depth_folder, depth) for depth in my_utils.os_listdir(depth_folder)]))
                # 读取第一张图像的长宽并按照render_scale进行缩放
                im = files[0]
                if im.endswith(".png") or im.endswith(".jpg"):
                    imH, imW = cv2.imread(im).shape[0:2]
                elif im.endswith(".npy"):
                    imH, imW = np.load(im).shape[0:2]
                else:
                    raise Exception("Invalid extension", im)
                imH = imH//self.settings["render_scale"]
                imW = imW//self.settings["render_scale"]


                print('Simulation: rain {}mm/hr {}'.format(intensity,name))
                # loading StreaksDBManager
                self.db = DBManager(streaks_path_pkl=sim_file, streaks_db=self.streaks_db,
                                    streaks_size=self.streaks_size)


                # 加载场景信息
                self.scene_renderer = SceneRenderer(scene_path=self.scene[sequence])
                self.scene_renderer.load_scene()

                # 加载雨纹数据
                # loading streaks from Streaks Database
                self.db.load_streak_database()
                self.db.load_streak_database_point()

                # 加载降雨粒子模拟文件，计算并存储对应的雨纹
                # creating drops based on the simulator file
                self.db.load_streaks_from_pkl(self.dataset, self.settings, [imW, imH], use_pickle=False, verbose=self.verbose)

                # 所有帧的雨纹
                frame_render_dict = list(self.db.streaks_simulator.values())

                f_start, f_end, f_step = self.frame_start, self.frame_end, self.frame_step
                # 取设定的结束帧和数据集图像数量的最小值
                f_end = len(files) if f_end is None else min(f_end, len(files))
                # 获取要渲染的帧的索引列表
                if self.frames:
                    # prone to go "boom", so we clip and remove 'wrong' ids
                    idx = np.unique(np.clip(self.frames, 0, f_end - 1)).tolist()
                else:
                    idx = list(range(f_start, f_end, f_step))  # to make it
                f_num = len(idx)
                sim_t0 = time.time()
                print("{} images".format(len(idx)))
                frames_exist_nb = 0
                # 渲染每一帧
                for f_idx, i in enumerate(idx):
                    image_file = files[i]           # 背景图像文件路径
                    depth_file = depth_files[i]     # 深度图像文件路径

                    f_name_idx = i

                    assert os.path.exists(image_file), "Image file {} does not exist".format(image_file)
                    assert os.path.exists(depth_file), "Depth file {} does not exist".format(depth_file)

                    # Ensure deterministic behavior
                    np.random.seed(f_name_idx)

                    frame_t0 = time.time()
                    # 粒子文件中帧数量小于图像数量则取模
                    frame = frame_render_dict[f_name_idx % len(frame_render_dict)]
                    file_name = os.path.split(image_file)[-1]
                    
                    # 输出路径：深度图，雨纹亮度颜色缓存，GBuffer
                    out_depth_path = os.path.join(out_dir, 'depth', '{}.png'.format(file_name[:-4]))
                    out_color_path = os.path.join(out_dir, 'buffer_color', '{}.png'.format(file_name[:-4]))
                    out_buffers_path = os.path.join(out_dir, 'buffers', '{}.npz'.format(file_name[:-4]))

                    frame_exists = os.path.exists(out_depth_path) or os.path.exists(out_color_path) or os.path.exists(out_buffers_path)
                    if frame_exists:
                        if self.conflict_strategy == "skip":
                            frames_exist_nb += 1
                            continue
                        elif self.conflict_strategy == "overwrite":
                            pass
                        else:
                            raise NotImplementedError


                    # flush should happens after a while
                    if self.verbose:
                        sys.stdout.write(
                            '\r' + my_utils.process_eta_str(process_t0, folder_idx, folders_num, folder_t0, sim_idx,
                                                            sim_num, sim_t0, f_idx,
                                                            f_num, frame_t0) + '                        ')

                    # 两张背景图，一张用于计算雨滴颜色，一张用于叠加雨滴缓存结果
                    # bg为原始背景图，保持不变
                    # two copies of bg because one is used for rain drop calculation
                    # and the other is changed after adding each drop
                    bg = cv2.imread(image_file) / 255.0

                    if self.settings["render_scale"] != 1:
                        bg = cv2.resize(bg, (int(bg.shape[1]//self.settings["render_scale"]), int(bg.shape[0]//self.settings["render_scale"])))

                    if USE_DEPTH_CULLING == 1:
                        # Depth map is used in weighting the luminance effect of the environment map on a single drop
                        if depth_file.endswith(".png"):
                            depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                            if depth is None:
                                print('Missing/Corrupted depth data (%s)' % depth_file)
                                continue
                            if depth.dtype == np.uint16:
                                depth = depth.astype(np.float32) / 65535.
                            else:
                                depth = depth.astype(np.float32) / 255.
                        elif depth_file.endswith(".npy"):
                            depth = np.load(depth_file)
                        else:
                            raise Exception("Invalid extension")

                        # Apply depth and render scale
                        depthHW = np.array([int((depth.shape[0] * self.settings["depth_scale"]) // self.settings["render_scale"]), int((depth.shape[1] * self.settings["depth_scale"]) // self.settings["render_scale"])])
                        if not np.all(depth.shape[:2] == depthHW):
                            depth = cv2.resize(depth, (depthHW[1], depthHW[0]))

                        assert (np.all(depth.shape[:2] <= bg.shape[:2])), "Depth cannot be larger than the image"

                        # Strategy to apply if RGB and Depth size mismatch
                        if not np.all(depth.shape[:2] == bg.shape[:2]):
                            # print("\nDepth {} size ({},{}) differs from image ({},{}). Will assume depth is crop centered.".format(image_file, depth.shape[0], depth.shape[1], bg.shape[0], bg.shape[1]))
                            bg = my_utils.crop_center(bg, depth.shape[0], depth.shape[1])
                    else:
                        # In that case, no need for depth, but it's still used down in the code, for more less nothing
                        depth = np.zeros((bg.shape[1], bg.shape[0]), np.float)


                    # 剔除不在图像内的雨纹
                    # Render only streaks inside the frame
                    streak_dict = frame.streaks
                    streak_dict = {keys: values for keys, values in streak_dict.items() if
                                   1 <= values.max_width < max(imH, imW) and
                                   1 <= values.length < max(imH, imW) and
                                   ((0 <= values.image_position_start[0] < imW
                                     and 0 <= values.image_position_start[1] < imH) or
                                    (0 <= values.image_position_end[0] < imW
                                     and 0 <= values.image_position_end[1] < imH))}


                    # 创建GBuffer
                    buffer_mask = np.zeros(shape=(bg.shape[0],bg.shape[1]),dtype=bool)
                    buffer_depth = depth[...,0].copy()
                    buffer_color = np.zeros(shape=(bg.shape[0],bg.shape[1],3))
                    gbuffers = ( buffer_mask,buffer_depth,buffer_color )  

                    streak_list = list(streak_dict.values())
                    drop_num = len(streak_list)
                    drop_process_t0 = time.time()
                    # 遍历渲染雨纹
                    for drop_idx, drop_dict in enumerate(streak_list):
                        if USE_DEPTH_CULLING == 1:
                            mean_x = (drop_dict.image_position_end[0] + drop_dict.image_position_start[0]) / 2
                            mean_y = (drop_dict.image_position_end[1] + drop_dict.image_position_start[1]) / 2
                            if drop_dict.linear_z >= depth[int(mean_y)][int(mean_x)][0]:
                                continue
                        # 计算GBuffer
                        gbuffers = self.compute_gbuffer(drop_dict,gbuffers,frame.exposure_time)

                        # Compute progress
                        avg_drop_time = (time.time() - drop_process_t0) / (drop_idx + 1)
                        if self.verbose or drop_idx == 0:
                            sys.stdout.write(
                                '\r' + my_utils.process_eta_str(process_t0, folder_idx, folders_num, folder_t0, sim_idx,
                                                                sim_num,
                                                                sim_t0, f_idx, f_num, frame_t0, drop_idx,
                                                                drop_num) + '\t\t' + '%.1fms /drop' % (
                                        1000. * avg_drop_time) + '       ')

                    # 用gbuffers重建世界坐标
                    ( buffer_mask,buffer_depth,buffer_color )  = gbuffers
                    buffer_pos_rec = self.scene_renderer.reconstructPoints(f_idx,gbuffers)
                    
                    # Create all output directories
                    # 创建输出路径
                    os.makedirs(os.path.dirname(out_depth_path), exist_ok=True)
                    os.makedirs(os.path.dirname(out_color_path), exist_ok=True)
                    os.makedirs(os.path.dirname(out_buffers_path), exist_ok=True)
                    

                    buffer_depth = np.dstack([buffer_depth,buffer_depth,buffer_depth])

                    # 保存结果
                    plt.imsave(out_depth_path, buffer_depth)
                    plt.imsave(out_color_path, buffer_color[...,0],cmap='gray')
                    buffer_pos_rec = buffer_pos_rec.astype('float32')
                    buffer_color = buffer_color.astype('float32')
                    np.savez(out_buffers_path,buffer_mask=buffer_mask,buffer_pos_rec=buffer_pos_rec,buffer_color=buffer_color)

                if frames_exist_nb > 0:
                    print("Skipped {}/{} already existing renderings".format(frames_exist_nb, len(idx)))

            print("\n\nEnd of the simulation")







