"""
判别器
"""


import tensorflow as tf
from tensorflow.keras import layers


# 判别器
class Discriminator(tf.keras.Model):
    def __init__(self, x_dim):
        super(Discriminator, self).__init__()
        self.x_dim = x_dim
        self.Line = tf.keras.Sequential([
            # 隐层
            layers.Dense(self.x_dim * 16, kernel_initializer='glorot_uniform'),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.5),
            
            layers.Dense(256, kernel_initializer='glorot_uniform'),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.5),

            layers.Dense(128, kernel_initializer='glorot_uniform'),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.5),

            layers.Dense(64, kernel_initializer='glorot_uniform'),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.5),
            
            # 输出层
            layers.Dense(1, kernel_initializer='glorot_uniform'),
            layers.Activation('sigmoid')
        ])

    def call(self, x, is_training=1):
        # 判别网络的输出
        out = self.Line(x, training=is_training)

        return out

