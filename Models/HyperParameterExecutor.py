import math

import tensorflow as tf
from CustomLosses import soft_dice_loss
from random import choice
import Config
import datetime
from DataLoader import DataLoader
from CommonLayers import XCoderBlockDefinition,ConvBlockDefinition, ModelDefinition
from ConvUNet import ConvUnetDefinition, create_unet
from VisionTransformer import TransformerDefinition, create_vit_model
import sys
import os
tf.keras.backend.set_floatx(Config.CURRENT_DTYPE)

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "../logs/fit" + timenow
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
model_path = "ModelInst/"+timenow+".keras"
def pick_randoms(args:list, count):
    result = []
    for _ in range(count):
        result.append(choice(args))
    return result


LOSSES = [tf.keras.losses.BinaryCrossentropy()]
OPTIMIZERS = ['adam']
KERNEL_SIZES_2D = [(2, 2), (3, 3), (5, 5), (7, 7), (1, 7), (7, 1)]
FILTER_COUNTS_2D = [4, 16, 64]
LAYER_COUNTS_2D = [2]

STRIDES = [1]

KERNEL_SIZES_3D = [(2, 2, 2), (3, 3, 3), (5, 5, 5), (7, 7, 7),
                        (7, 1, 1), (1, 7, 1), (1, 1, 7), (5, 5, 1), (5, 1, 5), (1, 5, 5)]
FILTER_COUNTS_3D = [2,3,4]
LAYER_COUNTS_3D = [1,2]
BOTTLENECK_FILTERS = [4]
XCODER_FACTORS = [2]
XCODER_BLOCKS = [3,4]

BATCH_NORM = [False]
DROPOUTS = [0.0,0.1,0.2,0.4]
ACTIVATIONS = ["leaky_relu","relu","sigmoid","gelu"]

TRANSFORMER_PATCH_SIZES = [4,8,16,32]
TRANSFORMER_INIT_FILTERS = [32,64]
TRANSFORMER_HEAD_C = [8,12,16]
TRANSFORMER_LATENT_LAYERS = [1,2,4,8]

LEARNING_RATES = [0.005,0.01,0.02,0.03,0.04]
def pick_conv_definition_2d():
    layer_c= choice(LAYER_COUNTS_2D)
    return ConvBlockDefinition(
        pick_randoms(KERNEL_SIZES_2D,layer_c),
        pick_randoms(STRIDES,layer_c),
        pick_randoms(FILTER_COUNTS_2D,layer_c),
        choice(BATCH_NORM),
        choice(DROPOUTS),
        choice(ACTIVATIONS)
    )
def pick_conv_definition_3d():
    layer_c= choice(LAYER_COUNTS_3D)
    return ConvBlockDefinition(
        pick_randoms(KERNEL_SIZES_3D,layer_c),
        pick_randoms(STRIDES,layer_c),
        pick_randoms(FILTER_COUNTS_3D,layer_c),
        choice(BATCH_NORM),
        choice(DROPOUTS),
        choice(ACTIVATIONS)
    )
def pick_encoder_definition(conv_block_c:int, factor:int):
    return XCoderBlockDefinition(
        factor,
        [pick_conv_definition_2d() for _ in range(conv_block_c)]
    )
def pick_decoder_definition(conv_block_c:int, factor:int):
    return XCoderBlockDefinition(
        factor,
        [pick_conv_definition_3d() for _ in range(conv_block_c)]
    )
def pick_model_definition():
    return ModelDefinition(choice(LOSSES),choice(OPTIMIZERS), choice(BOTTLENECK_FILTERS), choice(LEARNING_RATES))

def pick_unet_definition():
    xcoder_depth=choice(XCODER_BLOCKS)
    factor=choice(XCODER_FACTORS)
    return ConvUnetDefinition(
        pick_encoder_definition(xcoder_depth,factor),
        pick_decoder_definition(xcoder_depth,factor),
        pick_conv_definition_2d(),
        pick_model_definition()
    )
def pick_transformer_definition():
    patch_size = choice(TRANSFORMER_PATCH_SIZES)
    num_patches = Config.INPUT_IMAGE_SQ_RESOLUTION // patch_size
    factor=choice(XCODER_FACTORS)
    xcoder_depth=math.ceil(math.log(patch_size,factor))
    return TransformerDefinition(
        patch_size,
        choice(TRANSFORMER_INIT_FILTERS),
        choice(TRANSFORMER_HEAD_C),
        choice(TRANSFORMER_LATENT_LAYERS),
        choice(BOTTLENECK_FILTERS),
        pick_decoder_definition(xcoder_depth,factor),
        pick_model_definition()

    )

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

def save_cfg(title, definition):
    date=datetime.datetime.now()
    file_path = os.path.join(script_dir,f"{title}{date.month}{date.day}{date.hour}{date.minute}.def")
    with open(file_path, "w") as file:
        file.write(definition.__str__().__str__())

def main():
    if sys.argv[1] == "T":
        definition = pick_transformer_definition()
        save_cfg("T",definition)
        print("\n\nDEFINITION\n\n")
        print(definition.__str__())

        vit = create_vit_model(definition)
        vit.summary()
        print("\n\nDEFINITIONEND\n\n")
        print("\n\nTRAINING\n\n")
        loader = DataLoader(Config.X_FOLDER, Config.Y_FOLDER,16)
        val_loader = DataLoader(Config.VAL_X_FOLDER, Config.VAL_Y_FOLDER,16)
        vit.fit(loader, epochs=175, verbose=2, callbacks=[tensorboard_callback],validation_data=val_loader)
        vit.save(model_path)
        print("\n\nTRAININGEND\n\n")
    if sys.argv[1] == "C":
        definition = pick_unet_definition()
        save_cfg("C",definition)
        print("\n\nDEFINITION\n\n")
        print(definition.__str__())

        cnn = create_unet(definition)
        cnn.summary()
        print("\n\nDEFINITIONEND\n\n")
        print("\n\nTRAINING\n\n")
        loader = DataLoader(Config.X_FOLDER, Config.Y_FOLDER,16)
        val_loader = DataLoader(Config.VAL_X_FOLDER, Config.VAL_Y_FOLDER,16)
        cnn.fit(loader, epochs=150, verbose=2, callbacks=[tensorboard_callback],validation_data=val_loader)
        cnn.save(model_path)
        print("\n\nTRAININGEND\n\n")


if __name__ == "__main__":
    main()

