import tensorflow as tf
from keras.layers import Layer, Conv2D, Conv3D, Reshape, Conv1D
import math

from keras.src.layers import BatchNormalization, Dropout, Activation, MaxPooling2D, Conv3DTranspose, MultiHeadAttention, LayerNormalization
from keras.initializers import Initializer
import numpy as np
from keras.initializers import RandomUniform





class DivisionLayer(Layer):
    def __init__(self, y, **kwargs):
        super(DivisionLayer, self).__init__(**kwargs)
        self.y = y

    def call(self, inputs):
        return tf.divide(inputs, self.y)


# <unlearn-able>

class PrintLayer(Layer):
    def __init__(self, msg, print_input=False, **kwargs):
        super(PrintLayer, self).__init__(**kwargs)
        self.msg = msg
        self.print_input = print_input

    def call(self, inputs):
        print(self.msg)
        if self.print_input:
            print(inputs)
        return inputs


def conversion2d_3d(input_tensor, input_dim, target_filters):
    x = input_tensor
    x = Conv2D(filters=input_dim*target_filters,
        kernel_size=(1, 1),
        strides=1,
        padding='same',
        kernel_initializer=tf.keras.initializers.Ones(),
        bias_initializer=tf.keras.initializers.Zeros(),
        trainable=False
    )(x)
    x = DivisionLayer(input_dim)(x)
    x = Reshape((input_dim, input_dim, input_dim, target_filters))(x)
    return x


def conv_dimension_change(input_tensor, input_dim, output_dim):
    x = input_tensor
    gcd = math.gcd(input_dim, output_dim)
    filter_factor = output_dim // gcd
    stride_factor = input_dim // gcd
    flatten_shape = input_dim ** 3
    x = Reshape([flatten_shape, 1])(x)
    for dim in range(3):
        conv= Conv1D(filters=filter_factor, kernel_size=stride_factor, strides=stride_factor,
            kernel_initializer=tf.keras.initializers.Zeros(),trainable=False)
        x = conv(x)
        weights = conv.weights[0]
        shape = weights.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if i == k:
                        weights[i, j, k].assign(1)
        flatten_shape = flatten_shape // input_dim * output_dim
        x = Reshape([flatten_shape, 1])(x)
    x = Reshape([output_dim, output_dim, output_dim])(x)
    return x

# </unlearn-able>
class ConvBlockDefinition:
    def __init__(self, kernel_sizes:list, strides:list,  num_filters:list, batch_norm:bool, dropout:float,activation):
        self.activation = activation
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.num_filters = num_filters
        self.batch_norm = batch_norm

    def __str__(self):
        return self.__dict__
def conv_block2d(input_tensor, definition:ConvBlockDefinition):
    x = input_tensor
    for kernel_size, stride, filters in zip(definition.kernel_sizes, definition.strides, definition.num_filters):
        x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
        if definition.batch_norm:
            x = BatchNormalization()(x)
        x = Activation(definition.activation)(x)
        x = Dropout(definition.dropout)(x)
    return x

def conv_block3d(input_tensor, definition:ConvBlockDefinition):
    x = input_tensor
    for kernel_size, stride, filters in zip(definition.kernel_sizes, definition.strides, definition.num_filters):
        x = Conv3D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
        if definition.batch_norm:
            x = BatchNormalization()(x)
        x = Activation(definition.activation)(x)
        x = Dropout(definition.dropout)(x)
    return x

class XCoderBlockDefinition:
    def __init__(self, factor:int, definitions:list[ConvBlockDefinition]):
        self.conv_definitions = definitions
        self.factor = factor
    def __str__(self):
        return {"factor":self.factor, "conv_blocks": [d.__str__() for d in self.conv_definitions]}

def encoder_block2d(input_tensor, definition:XCoderBlockDefinition):
    x=input_tensor
    for conv_definition in definition.conv_definitions:
        x = conv_block2d(x, conv_definition)
        x = MaxPooling2D(definition.factor, strides=definition.factor, padding='same')(x)
    return x

def decoder_block3d(input_tensor, definition:XCoderBlockDefinition):
    x = input_tensor
    for conv_definition in definition.conv_definitions:
        x = conv_block3d(x, conv_definition)
        x = Conv3DTranspose(x.shape[-1], kernel_size=definition.factor, strides=definition.factor, padding='same')(x)
    return x


def multi_head_self_attention(x, num_heads=8):
    x = LayerNormalization(epsilon=1e-6)(x)
    attention_output = MultiHeadAttention(key_dim=64, num_heads=num_heads, dropout=0.1)(x, x)
    x = x + attention_output
    return x

class ModelDefinition:
    def __init__(self,loss,optimizer,bottle_neck_filters:int, lr:float):
        self.bottleneck_filters = bottle_neck_filters
        self.optimizer = optimizer
        self.loss = loss
        self.lr = lr

    def __str__(self):
        return self.__dict__