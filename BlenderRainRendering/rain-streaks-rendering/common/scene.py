import numpy as np
import taichi as ti
import taichi.math as tm

import json


# 场景渲染类
class SceneRenderer:
    def __init__(self, scene_path):
        self.scene_path = scene_path

    def load_scene(self):
        print("Loading Scene From JSON...")

        # 读取JSON文件
        with open(self.scene_path,"r") as f:
            data = json.load(f)
        
        self.cameras = []
        cameras_info = data["cameras"]
        for frame in cameras_info:
            camera_info = cameras_info[frame]
            viewMatrix = tm.mat4(np.array(camera_info["view_matrix"]).reshape((4,4)))
            viewMatrixInvert = tm.mat4(np.array(camera_info["view_matrix_invert"]).reshape((4,4)))
            projectionMatrix = tm.mat4(np.array(camera_info["projection_matrix"]).reshape((4,4)))
            projectionMatrixInvert = tm.mat4(np.array(camera_info["projection_matrix_invert"]).reshape((4,4)))
            self.cameras.append({
                "viewMatrix": viewMatrix,
                "viewMatrixInvert": viewMatrixInvert,
                "projectionMatrix": projectionMatrix,
                "projectionMatrixInvert": projectionMatrixInvert,
                "clip_start": camera_info["clip_start"],
                "clip_end": camera_info["clip_end"],
                "fov": camera_info["fov"]
            })
        
        print("Load Scene Done.")


    # 重建雨纹世界坐标
    def reconstructPoints(self,f_idx,gbuffers):
        ( buffer_mask,buffer_depth,_)  = gbuffers
        camera = self.cameras[f_idx]

        image_size = buffer_mask.shape[:2]
        ti_output = ti.Vector.field(3, dtype=float, shape=(image_size[0],image_size[1],4))
        ti_depth = ti.field(dtype=float, shape=image_size)
        ti_depth.from_numpy(buffer_depth)

        offsets = ti.field(dtype=tm.vec2,shape=4)
        offsets.from_numpy(np.array([[0,0],[1,0],[1,1],[0,1]]))

        reconstruct_world_pos(ti_depth,ti_output,image_size,camera["clip_end"],offsets,
                              camera["projectionMatrixInvert"],camera["viewMatrixInvert"])
        buffer_output = ti_output.to_numpy()
        
        del ti_depth
        del ti_output

        return buffer_output


    @staticmethod
    def warping_points(drop, drop_texture, image_width, image_height):
        x0 = round(drop.image_position_start[0])
        x1 = round(drop.image_position_end[0])
        y0 = round(drop.image_position_start[1])
        y1 = round(drop.image_position_end[1])
        d0 = np.floor(drop.image_diameter_start)
        d1 = np.floor(drop.image_diameter_end)

        minx = max(min(x0, x1), 0)
        miny = max(min(y0, y1), 0)
        maxx = min(max(x0 + d0, x1 + d1), image_width)
        maxy = min(max(y0, y1), image_height)

        # to prevent singularity of pers matrix
        epsilon = 0.001

        p1 = np.float32([
            [0, 0],
            [drop_texture.shape[1], 0],
            [drop_texture.shape[1], drop_texture.shape[0]],
            [0, drop_texture.shape[0]]])

        p2 = np.float32([
            [x0 - minx, y0 - miny],
            [x0 - minx + d0, y0 - miny],
            [x1 - minx + d1 + epsilon, y1 - miny],
            [x1 - minx + epsilon, y1 - miny]])

        return p1, p2, np.array([maxx, maxy]), np.array([minx, miny])

# 根据深度图重建世界坐标
@ti.kernel
def reconstruct_world_pos(ti_depth: ti.template(), ti_output: ti.template(),
                          image_size: tm.ivec2, clip_end: float, offsets: ti.template(),
                          projectionMatrixInvert: tm.mat4, viewMatrixInvert: tm.mat4):

    for y,x,pi in ti.ndrange(image_size[0],image_size[1],4):
        depth = ti_depth[y,x]
        offset = offsets[pi]
        ndc_far = ((tm.vec2(x,y) + offset) / image_size.yx) * 2.0 - 1.0
        clip_far = tm.vec3(ndc_far,1.0) * clip_end
        view_far = projectionMatrixInvert @ clip_far.xyzz
        view_rec = view_far * depth
        view_rec.w = 1
        world_rec = viewMatrixInvert @ view_rec
        ti_output[y, x, pi] = world_rec.xyz