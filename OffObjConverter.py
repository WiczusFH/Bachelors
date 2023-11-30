import trimesh
from PrincetonObjectIterator import Iterator
from Models import Config


def convert(source_path, target_path, object_name):
    trimesh\
        .load(source_path)\
        .export(target_path.replace(Config.OFF_FORMAT_NAME, Config.OBJ_FORMAT_NAME))


def is_obj_valid(object_name):
    return object_name.endswith(Config.OFF_FORMAT_NAME)


iterator = Iterator(Config.OFF_FOLDER, Config.MODEL_FOLDER)
iterator.execute(convert, is_obj_valid)
