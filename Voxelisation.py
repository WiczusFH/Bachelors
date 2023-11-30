import numpy as np
import trimesh
from PrincetonObjectIterator import Iterator, is_index_valid
from Models import Config

"""Increment index to last index if progress was interrupted"""
STARTING_INDEX = 0

def action(source_path, target_path, object_name):
    voxel_data = trimesh \
        .load(source_path) \
        .voxelized(pitch=Config.VOXEL_RESOLUTION) \
        .matrix
    paddings = [Config.VOXELIZED_SQ_RESOLUTION - voxel_data.shape[i] for i in range(3)]
    padding = [(padding // 2, padding // 2 + padding % 2) for padding in paddings]

    padded_voxel_data = np.pad(voxel_data,
        padding,
        mode='constant',
        constant_values=0)
    np.save(f"{target_path[:-len(Config.OBJ_FORMAT_NAME)]}{Config.NPY_FORMAT_NAME}", padded_voxel_data)
    print(f"Voxelized: {object_name}")


def is_obj_valid(object_name):
    if object_name.endswith(Config.OBJ_FORMAT_NAME):
        return is_index_valid(object_name, Config.OBJ_FORMAT_NAME, STARTING_INDEX)
    return False


iterator = Iterator(Config.RENDERED_FOlDER, Config.Y_FOLDER, flatten=True)
iterator.execute(action, is_obj_valid)
