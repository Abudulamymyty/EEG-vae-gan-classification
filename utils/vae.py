"""
编码器和解码器类
根据均值和方差进行采样得到z
"""

import tensorflow as tf
from tensorflow.keras import layers


# 编码器
class Encoder(tf.keras.Model):
    def __init__(self, enc_dim):
        super(Encoder, self).__init__()
        self.enc_dim = enc_dim * 2
        # Sequential()方法是一个容器，描述了神经网络的网络结构，在Sequential()的输入参数中描述从输入层到输出层的网络结构
        self.Encoder0 = tf.keras.Sequential([
            
            layers.Dense(128),  # 全连接层，units=128是输出节点数
            layers.LeakyReLU(0.2),  # 使用LeakyReLU作为激活函数
            layers.Reshape((16, 8)),  # 形状重塑层，将128重塑为16*8

            layers.Conv1D(16, 3, padding='same'),  # 一维卷积层，输出空间维度为16，卷积核大小为3，并用0填充剩余值，保持输入输出尺寸相同

            layers.Conv1D(32, 3),  # 一维卷积层，输出空间维度为32，卷积核大小为3，padding默认为valid，没有填充

            layers.Flatten(),  # 压平层，把多维的输入一维化，常用在从卷积层到全连接层的过渡
            
            layers.Dense(128),  # 全连接层，units=128是输出节点数
            layers.Dropout(0.5),  # 随机失活，有利于防止过拟合
            layers.LeakyReLU(0.2)  # 使用LeakyReLU作为激活函数
        ])
        self.Encoder1 = tf.keras.Sequential([
            layers.Dense(self.enc_dim, activation='tanh'),
        ])

    # 具体执行
    def call(self, x, is_training=1):
        temp = self.Encoder0(x, training=is_training)
        out = self.Encoder1(temp, training=is_training)

        return out


# 解码器
class Decoder(tf.keras.Model):
    def __init__(self, x_dim):
        super(Decoder, self).__init__()
        self.x_dim = x_dim

        self.Decoder = tf.keras.Sequential([

            layers.Dense(128),
            layers.LeakyReLU(0.2),
            layers.Reshape((8, 16)),

            layers.Conv1D(32, 3, padding='same'),
            
            layers.Conv1D(16, 3),
            layers.Dropout(0.5),
            layers.Flatten(),
            
            layers.Dense(self.x_dim, activation='tanh'),
        ])

    def call(self, x, is_training=1):
        out = self.Decoder(x, training=is_training)
        return out


def sample_z(mean, stddev):
    """
    根据均值和标准差进行采样得到z
    具体做法：先得到z的标准正态分布，再根据均值和标准差进行变换得到z
    :param : mean : 均值
    :param : stddev : 标准差
    """
    # # 标准正态分布生成器
    # std_normal_distribution_init = tf.random_normal_initializer(stddev=1.0) 
    # # 得到z的标准正态分布std_z
    # std_z = std_normal_distribution_init(shape=mean.shape)
    
    # # 变换 
    # z = mean + std_z * stddev

    # return z

    eps_init = tf.random_normal_initializer()
    eps = eps_init(shape=mean.shape)

    return mean + eps * tf.exp(stddev)


def get_vae_prior_loss(mean, stddec):
    """
    计算VAE的先验损失（Dkl）
    :param : mean : 隐编码分布的均值
    :param : stddec : 隐编码分布的标准差，改为标准差的对数

    :return : vae_prior_loss : VAE的先验损失
    """
    # vae_prior_loss = - tf.reduce_mean(0.5 * (tf.math.log(tf.square(stddec)) - tf.square(stddec) - tf.square(mean) + 1))
    vae_prior_loss = - tf.reduce_mean(0.5 * (2 * stddec - tf.square(tf.exp(stddec)) - tf.square(mean) + 1))
    return vae_prior_loss


def get_vae_likelihood_loss(original_x, generated_x):
    """
    计算VAE的似然损失
    :param : original_x : 原样本
    :param : generated_x : 经过VAE网络生成的的样本

    :return : vae_likelihood_loss : VAE的似然损失
    """
    vae_likelihood_loss = tf.reduce_mean(tf.square(original_x - generated_x))
    return vae_likelihood_loss


def get_vae_loss(mean, stddec, original_x, generated_x, alpha_1=1.5, alpha_2=0.1):
    """
    计算VAE的损失
    
    :param : mean : 隐编码分布的均值
    :param : stddec : 隐编码分布的标准差
    :param : original_x : 原样本
    :param : generated_x : 经过VAE网络生成的的样本
    :param : alpha_1 :   default = 1.5
    :param : alpha_2 :   default = 0.1

    :return : loss : 计算vae损失
    """
    vae_prior_loss = get_vae_prior_loss(mean=mean, stddec=stddec)
    vae_likelihood_loss = get_vae_likelihood_loss(original_x=original_x, generated_x=generated_x)
    
    vae_loss = (alpha_1 * vae_prior_loss) + (alpha_2 * vae_likelihood_loss)
    
    return vae_loss

