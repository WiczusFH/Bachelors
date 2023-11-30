import bpy
import random
import math
from PrincetonObjectIterator import Iterator, add_index2path
from Models import Config

RENDER_VARIATIONS = 1

# Camera
def make_camera():
    bpy.context.scene.render.resolution_x = Config.INPUT_IMAGE_SQ_RESOLUTION
    bpy.context.scene.render.resolution_y = Config.INPUT_IMAGE_SQ_RESOLUTION
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=Config.CAMERA_LOCATION)
    camera = bpy.context.object
    camera.rotation_euler.x += math.radians(90)
    camera.data.sensor_width = Config.INPUT_IMAGE_SQ_RESOLUTION
    camera.data.sensor_height = Config.INPUT_IMAGE_SQ_RESOLUTION
    camera.data.lens = Config.INPUT_IMAGE_SQ_RESOLUTION
    bpy.context.scene.camera = camera


def make_sun_rand(energy=1.0, min_whiteness=0.7):
    bpy.ops.object.light_add(type='SUN')
    sun = bpy.context.object
    sun.rotation_euler = (random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi))
    sun.data.color = (random.uniform(min_whiteness, 1), random.uniform(min_whiteness, 1), random.uniform(min_whiteness, 1))
    sun.data.energy = energy


def make_obj_rand(path, output_full_path):
    # Import the object into the scene
    bpy.ops.import_scene.obj(filepath=path)
    imported_object = bpy.context.selected_objects[0]
    imported_object.select_set(True)
    # Scaling to fit into the camera
    size_x, size_y, size_z = imported_object.dimensions
    downscale_factor = 1 / max(size_x, size_y, size_z)
    imported_object.scale.x = downscale_factor
    imported_object.scale.y = downscale_factor
    imported_object.scale.z = downscale_factor
    # Random rotation to ensure we have informations about all sides
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')
    bpy.ops.transform.rotate(value=random.uniform(0, 2 * math.pi), orient_axis='X')
    bpy.ops.transform.rotate(value=random.uniform(0, 2 * math.pi), orient_axis='Y')
    bpy.ops.transform.rotate(value=random.uniform(0, 2 * math.pi), orient_axis='Z')
    imported_object.location = (0.0, 0.0, 0.75)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # Add further rescaling to simulate varying distance
    random_rescaling = random.uniform(0.6, 1.1)
    imported_object.scale.x = random_rescaling
    imported_object.scale.y = random_rescaling
    imported_object.scale.z = random_rescaling
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # Shade smooth
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.shade_smooth()

    bpy.ops.export_scene.obj(filepath=f"{output_full_path}", use_selection=True, use_materials=False)


def render(output_full_path):
    bpy.ops.render.render(write_still=True)
    render_output = bpy.context.scene.render.filepath
    bpy.context.scene.render.filepath = render_output

    bpy.data.images['Render Result'].save_render(f"{output_full_path}")


def del_all():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def run(source_path, target_path, object_name):
    for i in range(RENDER_VARIATIONS):
        make_camera()
        make_sun_rand(energy=3.5)
        make_sun_rand(energy=0.5)
        make_sun_rand(energy=0.1)
        indexed_target_path = add_index2path(target_path, Config.OBJ_FORMAT_NAME)
        make_obj_rand(source_path, f"{indexed_target_path}{Config.OBJ_FORMAT_NAME}")
        render(f"{indexed_target_path}{Config.PNG_FORMAT_NAME}")
        del_all()


def is_obj_valid(object_name):
    return object_name.endswith(Config.OBJ_FORMAT_NAME)


iterator = Iterator(Config.MODEL_FOLDER, Config.RENDERED_FOlDER)
iterator.execute(run, is_obj_valid)
