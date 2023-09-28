import tensorflow.compat.v1 as tf
keras = tf.keras
layers = tf.keras.layers
Dense = layers.Dense
Conv2D = layers.Conv2D
Conv1D = layers.Conv1D
GlobalAveragePooling2D = layers.GlobalAveragePooling2D
Input = layers.Input
Reshape = layers.Reshape
Activation = layers.Activation
multiply = layers.Multiply
Model = tf.keras.Model
import math

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import Model, layers
# import math

def eca_block(inputs, b=1, gama=2):
    # 输入特征图的通道数
    in_channel = int(inputs.shape[-1])    #

    # 根据公式计算自适应卷积核大小
    kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))

    # 如果卷积核大小是偶数，就使用它
    if kernel_size % 2:
        kernel_size = kernel_size

    # 如果卷积核大小是奇数就变成偶数
    else:
        kernel_size = kernel_size + 1

    # [h,w,c]==>[None,c] 全局平均池化
    x = layers.GlobalAveragePooling2D()(inputs)

    # [None,c]==>[c,1]
    x = layers.Reshape(target_shape=(in_channel, 1))(x)

    # [c,1]==>[c,1]
    x = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)

    # sigmoid激活
    x = tf.nn.sigmoid(x)

    # [c,1]==>[1,1,c]
    x = layers.Reshape((1, 1, in_channel))(x)

    # 结果和输入相乘
    outputs = layers.multiply([inputs, x])

    return outputs


# 验证ECA注意力机制
# if __name__ == '__main__':
#     # 构造输入层
#     inputs = keras.Input(shape=[26, 26, 512])
#     x = eca_block(inputs)  # 接收ECA输出结果
#
#     model = Model(inputs, x)  # 构造模型
#     model.summary()  # 查看网络架构