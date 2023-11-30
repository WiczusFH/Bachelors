INPUT_IMAGE_SQ_RESOLUTION = 256
CHANNEL_COUNT=3
INPUT_IMAGE_SHAPE = (INPUT_IMAGE_SQ_RESOLUTION, INPUT_IMAGE_SQ_RESOLUTION, CHANNEL_COUNT)
# Max size of the object is 1.1 (this is defined in the blender script)
# For non-rotated objects 1.1*128 ~ 142
# For rotated objects (2*128^2)^0.5*1.1 ~ 200
# 228 is a rounded composite of 2^n that is certain to fit the model
VOXELIZED_SQ_RESOLUTION = 192
VOXELIZED_SHAPE = (VOXELIZED_SQ_RESOLUTION,VOXELIZED_SQ_RESOLUTION,VOXELIZED_SQ_RESOLUTION)
VOXEL_RESOLUTION = 1 / 128
CAMERA_LOCATION = (0, -1.5, 0.75)

CURRENT_DTYPE="float32"

OBJ_FORMAT_NAME = ".obj"
NPY_FORMAT_NAME = ".npy"
PNG_FORMAT_NAME = ".png"
OFF_FORMAT_NAME = ".off"

RENDERED_FOlDER= "D:\\bch data\\3DModelsRendered"
MODEL_FOLDER = 'D:\\bch data\\3DModels'
OFF_FOLDER = 'D:\\bch data\\3DModelsOff'

Y_FOLDER = "../train_y"
X_FOLDER = "../train_x"
VAL_Y_FOLDER = "../train_y_val"
VAL_X_FOLDER = "../train_x_val"

