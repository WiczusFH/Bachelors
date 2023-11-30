from tensorflow import keras
from keras.layers import Embedding, Dense
from keras import Model
import Config
from CommonLayers import *


class TransformerDefinition:
    def __init__(self,
                 patch_size: int,
                 initial_filters: int,
                 num_heads: int,
                 num_layers: int,
                 bottle_neck_filters:int,
                 decoder_definition: XCoderBlockDefinition,
                 model_definition: ModelDefinition):
        self.bottle_neck_filters = bottle_neck_filters
        self.initial_filters = initial_filters
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.decoder_definition = decoder_definition
        self.model_definition = model_definition

    def __str__(self):
        return {"bottle_neck":self.bottle_neck_filters.__str__(),
                "i_filter_c":self.initial_filters.__str__(),
                "patch_size":self.patch_size.__str__(),
                "head_c":self.num_heads.__str__(),
                "layer_c":self.num_layers.__str__(),
                "decoder":self.decoder_definition.__str__(),
                "model":self.model_definition.__str__()}

def create_vit_model(definition:TransformerDefinition):
    x = keras.Input(shape=Config.INPUT_IMAGE_SHAPE)
    patch_embeddings = Conv2D(filters=definition.initial_filters, kernel_size=definition.patch_size, strides=definition.patch_size,padding='same')(x)

    patch_c = Config.INPUT_IMAGE_SQ_RESOLUTION // definition.patch_size
    patch_embeddings = Reshape((patch_c, patch_c, definition.initial_filters))(patch_embeddings)

    positional_embeddings = Embedding(input_dim=patch_c ** 2, output_dim=definition.initial_filters)(tf.range(patch_c))
    y = patch_embeddings + positional_embeddings

    for _ in range(definition.num_layers):
        y = MultiHeadAttention(
            num_heads=definition.num_heads,
            key_dim=definition.initial_filters // definition.num_heads,
            dropout=0.1)(y, y)
        y = LayerNormalization(epsilon=1e-6)(y)
        y = Dense(units=definition.initial_filters, activation="relu")(y)
        y = LayerNormalization(epsilon=1e-6)(y)

    y = conversion2d_3d(y, patch_c, definition.bottle_neck_filters)
    y = decoder_block3d(y, definition.decoder_definition)
    input_dim = y.shape[1]

    y = Conv3D(filters=1,kernel_size=1,strides=1)(y)
    y = conv_dimension_change(y, input_dim, output_dim=Config.VOXELIZED_SQ_RESOLUTION)
    y = Activation('sigmoid')(y)

    model = Model(x, y)
    model.compile(optimizer=definition.model_definition.optimizer,
        loss=definition.model_definition.loss)
    return model


# Example usage:

