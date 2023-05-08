"""
映射网络的实现
"""


import tensorflow as tf
from tensorflow.keras import layers


class Map_Net(tf.keras.Model):
    """
    映射网络

    把多数类样本的隐编码映射为少数类样本的隐编码
    """
    def __init__(self, enc_dim):
        self.enc_dim = enc_dim * 2
        super(Map_Net, self).__init__()
        self.Map_Net0 = tf.keras.Sequential([
            
            layers.Dense(128),  # 全连接层，units=128是输出节点数
            layers.Dropout(0.5),  # 随机失活，有利于防止过拟合
            layers.LeakyReLU(0.2),  # 使用LeakyReLU作为激活函数
            
            layers.Dense(256),  # 全连接层，units=128是输出节点数
            layers.Dropout(0.5),  # 随机失活，有利于防止过拟合
            layers.LeakyReLU(0.2),  # 使用LeakyReLU作为激活函数

            layers.Dense(512),  # 全连接层，units=128是输出节点数
            layers.Dropout(0.5),  # 随机失活，有利于防止过拟合
            layers.LeakyReLU(0.2),  # 使用LeakyReLU作为激活函数

            layers.Dense(512),  # 全连接层，units=128是输出节点数
            layers.Dropout(0.5),  # 随机失活，有利于防止过拟合
            layers.LeakyReLU(0.2),  # 使用LeakyReLU作为激活函数

            layers.Dense(256),  # 全连接层，units=128是输出节点数
            layers.Dropout(0.5),  # 随机失活，有利于防止过拟合
            layers.LeakyReLU(0.2),  # 使用LeakyReLU作为激活函数

            layers.Dense(self.enc_dim),  # 全连接层，units=128是输出节点数
            layers.Dropout(0.5),  # 随机失活，有利于防止过拟合
            layers.LeakyReLU(0.2)  # 使用LeakyReLU作为激活函数
        ])
        self.Map_Net1 = tf.keras.Sequential([
            layers.Dense(self.enc_dim, activation='tanh'),
        ])
    def call(self, input_enc, is_training=1):
        """
        :param : input_enc : 输入的多数类样本的编码
        :return : output_enc : 得到的少数类样本的编码
        """
        temp = self.Map_Net0(input_enc, training=is_training)
        output_enc = self.Map_Net1(temp, training=is_training)

        return output_enc

