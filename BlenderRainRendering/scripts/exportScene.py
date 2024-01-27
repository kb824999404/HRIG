import bpy, mathutils
import os, platform
import json


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


def property_to_str(value):
    if isinstance(value, list) or isinstance(value, (mathutils.Vector, mathutils.Color, mathutils.Euler)):
        output = []
        for v in value:
            output.append(v)
        return output
    elif isinstance(value, mathutils.Matrix):
        output = []
        for row in value:
            for v in row:
                output.append(v)
        return output
    else:
        return value



def get_camera_info():

    camera_info = {}
    
    camera_info["name"] = property_to_str(camera.name)
    camera_info["type"] = property_to_str(camera.data.type)
    camera_info["location"] =property_to_str(camera.location)
    camera_info["rotation"] = property_to_str(camera.rotation_euler)
    camera_info["lens"] = property_to_str(camera.data.lens)
    camera_info["fov"] = property_to_str(camera.data.angle)
    camera_info["lens_unit"] = property_to_str(camera.data.lens_unit)
    camera_info["clip_start"] = property_to_str(camera.data.clip_start)
    camera_info["clip_end"] = property_to_str(camera.data.clip_end)
    if camera.data.type == 'ORTHO':
        camera_info["ortho_scale"] = property_to_str(camera.data.ortho_scale)
    camera_info["sensor_fit"] = property_to_str(camera.data.sensor_fit)
    camera_info["sensor_width"] = property_to_str(camera.data.sensor_width)
    camera_info["sensor_height"] = property_to_str(camera.data.sensor_height)


    depsgraph = bpy.context.evaluated_depsgraph_get()
    projection_matrix = camera.calc_matrix_camera(depsgraph,
                                                    x=render.resolution_x,y=render.resolution_y,
                                                    scale_x=render.pixel_aspect_x,scale_y=render.pixel_aspect_y)
    view_matrix = camera.matrix_world.inverted()
    camera_info["projection_matrix"] = property_to_str(projection_matrix)
    camera_info["projection_matrix_invert"] = property_to_str(projection_matrix.inverted())
    camera_info["view_matrix"] = property_to_str(view_matrix)
    camera_info["view_matrix_invert"] = property_to_str(camera.matrix_world)


    return camera_info

def get_render_info():
    render_info = {}
    render_info["engine"] = property_to_str(render.engine)
    render_info["fps"] = property_to_str(render.fps)
    render_info["resolution_x"] = property_to_str(render.resolution_x)
    render_info["resolution_y"] = property_to_str(render.resolution_y)
    render_info["pixel_aspect_x"] = property_to_str(render.pixel_aspect_x)
    render_info["pixel_aspect_y"] = property_to_str(render.pixel_aspect_y)

    return render_info

if __name__ == "__main__":
    output_root = os.getcwd()

    context = bpy.context
    scene = context.scene
    render = scene.render
    camera = scene.camera

    frame_start = scene.frame_start
    frame_end = scene.frame_end
    frame_step = scene.frame_step

    result = {}
    result["frame_start"] = scene.frame_start
    result["frame_end"] = scene.frame_end
    result["frame_step"] = scene.frame_step

    result["render"] = get_render_info()

    cameras_info = {}
    for f_idx,frame in enumerate(range(frame_start,frame_end+1,frame_step)):
        print("Processing frame {} ...".format(frame))
        scene.frame_set(frame)
        cameras_info[frame] = get_camera_info()
    result["cameras"] = cameras_info

    path = os.path.join(output_root,"scene_info.json")
    with open(path,"w") as f:
        json.dump(result,f)