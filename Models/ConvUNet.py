import keras.initializers
from keras.models import Model
from keras.layers import Input
import Config
from CommonLayers import *


class ConvUnetDefinition:
    def __init__(self, encoder_definition: XCoderBlockDefinition,
                 decoder_definition: XCoderBlockDefinition,
                 latent_definition: ConvBlockDefinition,
                 model_definition: ModelDefinition):
        self.encoder_definition = encoder_definition
        self.decoder_definition = decoder_definition
        self.latent_definition = latent_definition
        self.model_definition = model_definition

    def __str__(self):
        return {"encoder":self.encoder_definition.__str__(),
                "decoder":self.decoder_definition.__str__(),
                "latent":self.latent_definition.__str__(),
                "model":self.model_definition.__str__()}

def create_unet(definition: ConvUnetDefinition):
    x = Input(Config.INPUT_IMAGE_SHAPE)
    y = BatchNormalization()(x)
    y = encoder_block2d(y, definition.encoder_definition)

    y = conv_block2d(y, definition.latent_definition)
    input_dim = y.shape[1]
    y = conversion2d_3d(y, input_dim, definition.model_definition.bottleneck_filters)

    y = decoder_block3d(y, definition.decoder_definition)

    input_dim = y.shape[1]
    y = Conv3D(filters=1,kernel_size=1,strides=1, kernel_initializer=keras.initializers.Ones())(y)
    y = conv_dimension_change(y, input_dim, output_dim=Config.VOXELIZED_SQ_RESOLUTION)
    y = Activation('sigmoid')(y)
    model = Model(inputs=x, outputs=y)
    model.compile(optimizer=definition.model_definition.optimizer,
        loss=definition.model_definition.loss)
    return model
