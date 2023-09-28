# 学生：李屹
# 开发时间：2023/5/20 21:08
# import tensorflow.compat.v1 as tf
# from tensorflow.keras import layers, Model, Input
import tensorflow.compat.v1 as tf
keras = tf.keras
layers = tf.keras.layers
Dense = layers.Dense
Conv2D = layers.Conv2D
GlobalAveragePooling2D = layers.GlobalAveragePooling2D
Input = layers.Input
Reshape = layers.Reshape
Activation = layers.Activation
Multiply = layers.Multiply
Model = tf.keras.models

# se注意力机制
def se_block(inputs, ratio=4):  # ratio代表第一个全连接层下降通道数的系数
    # 获取输入特征图的通道数
    in_channel = inputs.shape[-1]

    # 全局平均池化[h,w,c]==>[None,c]
    x = layers.GlobalAveragePooling2D()(inputs)
    # [None,c]==>[1,1,c]
    x = layers.Reshape(target_shape=(1, 1, in_channel))(x)
    # [1,1,c]==>[1,1,c/4]
    x = layers.Dense(in_channel // ratio)(x)  # 全连接下降通道数
    # relu激活
    x = tf.nn.relu(x)
    # [1,1,c/4]==>[1,1,c]
    x = layers.Dense(in_channel)(x)  # 全连接上升通道数
    # sigmoid激活，权重归一化
    x = tf.nn.sigmoid(x)
    # [h,w,c]*[1,1,c]==>[h,w,c]
    outputs = layers.multiply([inputs, x])  # 归一化权重和原输入特征图逐通道相乘

    return outputs

