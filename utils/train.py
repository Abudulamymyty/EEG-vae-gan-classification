"""
smsg_pro_pro训练
"""

import numpy as np
import tensorflow as tf

from utils.data_process import batch_list_to_df
from utils.gan import Discriminator
from utils.generate_new_samples import generate_new_1_sample
from utils.get_performance import get_three_methods_performance_df
from utils.map_net import Map_Net
from utils.vae import Encoder, Decoder, sample_z, get_vae_likelihood_loss, get_vae_loss


def init_net(x_dim, enc_dim):
    """
    初始化SMSG_PRO的网络结构
    enc    dec
    map_net
    dis

    :param : x_dim : 样本的维度
    :param : enc_dim : 编码器编码的维度

    :return : smsg_net_dict = {'enc':enc, 'dec':dec, 'map_net':map_net, 'dis':dis}
    """
    enc = Encoder(enc_dim)  # 编码器
    dec = Decoder(x_dim)  # 解码器
    map_net = Map_Net(enc_dim)  # 映射网络
    dis = Discriminator(x_dim)  # 对样本进行判别的判别器
    dis_map = Discriminator(x_dim=enc_dim)  # 对隐编码进行判别的判别器

    smsg_net_dict = {'enc': enc, 'dec': dec, 'map_net': map_net, 'dis': dis, 'dis_map': dis_map}
    return smsg_net_dict


def init_optimizers(learning_rate=2e-4, beta_1=0.5):
    """
    初始化上述网络的优化器
    网络优化策略设置，优化器为Adam(学习率为2e-4，beta_1为0.5)
    :param : learning_rate default = 2e-4
    :param : beta_1 default = 0.5

    :return : optimizers_dict = {'optimizer_enc':optimizer_enc, 'optimizer_dec':optimizer_dec,
                                 'optimizer_map_net':optimizer_map_net, 'optimizer_dis':optimizer_dis, }

    """

    optimizer_enc = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)
    optimizer_dec = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)
    optimizer_map_net = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)
    optimizer_dis = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)
    optimizer_dis_map = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)

    optimizers_dict = {'optimizer_enc': optimizer_enc, 'optimizer_dec': optimizer_dec,
                       'optimizer_map_net': optimizer_map_net, 'optimizer_dis': optimizer_dis,
                       'optimizer_dis_map': optimizer_dis_map}

    return optimizers_dict


def train(smsg_net_dict, optimizers_dict, train_batch, val_batch, vae_loss_paramaters_dict, columns, epochs=300):
    """
    一次交叉验证的训练过程，同时将每训练一次batch的相关的损失写入到文件中
    :param : smsg_net_dict : SMSG_PRO网络结构字典
    :param : optimizers_dict : 所有网络优化器字典
    :param : train_batch : 一次交叉验证中训练集的batch，包括多数类样本和少数类样本的batches
    :param : val_batch : 一次交叉验证中验证集的batch，包括多数类样本和少数类样本的batches
    :param : vae_loss_paramaters_dict : 计算VAE损失的参数的字典 {'alpha_1':alpha_1_value, 'alpha_2':alpha_2_value}
    :param : columns : 列名，用于构造新样本时df的构造，检验后验崩塌
    :param : epochs : 训练轮数
    
    :return : optimized_smsg_net_dict : 优化过的SMSG的网络结构
    """

    # 初始化损失的pandas.DataFrame
    # loss_df = pd.DataFrame(columns = ['epoch', 'enc0_loss', 'dec0_loss', 'enc1_loss', 'dec1_loss', 'dis0_loss', 'dis1_loss'])

    # 获取SMSG_PRO中所有的网络以及对应的优化器
    enc = smsg_net_dict.get('enc')
    dec = smsg_net_dict.get('dec')
    map_net = smsg_net_dict.get('map_net')
    dis = smsg_net_dict.get('dis')
    dis_map = smsg_net_dict.get('dis_map')
    optimizer_enc = optimizers_dict.get('optimizer_enc')
    optimizer_dec = optimizers_dict.get('optimizer_dec')
    optimizer_map_net = optimizers_dict.get('optimizer_map_net')
    optimizer_dis = optimizers_dict.get('optimizer_dis')
    optimizer_dis_map = optimizers_dict.get('optimizer_dis_map')

    # 获取计算VAE损失的参数alpha_1和alpha_2
    alpha_1 = vae_loss_paramaters_dict.get('alpha_1')
    alpha_2 = vae_loss_paramaters_dict.get('alpha_2')

    val_performance_list = []
    optimized_smsg_net_dict_list = []

    # 综合训练
    for epoch in range(epochs):
        # 根据train_batch获取每一次训练的多数类样本batch和少数类样本batch
        for data_0_batch, data_1_batch in zip(train_batch[0], train_batch[1]):
            x_0, y_0 = data_0_batch
            x_1, y_1 = data_1_batch

            # 类型转换，将x_0, x_1中数据类型转为tf.float32
            x_0 = tf.cast(x_0, dtype=tf.float32)
            x_1 = tf.cast(x_1, dtype=tf.float32)

            # 使用自动微分机制进行训练
            with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as map_net_tape, tf.GradientTape() as dis_tape, tf.GradientTape() as dis_map_tape:
                # 多数类样本重构
                mean_and_stddec_0 = enc(x_0)  # 使用编码器对多数类样本进行编码
                mean_0, stddec_0 = tf.split(mean_and_stddec_0, 2, 1)  # 将1x128拆分为2x64，获取分布的两个参数
                z_0 = sample_z(mean_0, stddec_0)  # 根据分布的两个参数得到分布然后采样取值
                x_0_0 = dec(z_0)  # 使用解码器0生成多数类样本，即由多数类样本重构得到的多数类样本

                # 多数类样本迁移得到迁移少数类样本
                mean_and_stddec_0_1 = map_net(mean_and_stddec_0)  # 多数类样本的编码经过映射网络得到少数类样本的迁移编码
                mean_0_1, stddec_0_1 = tf.split(mean_and_stddec_0_1, 2, 1)  # 获取分布的两个参数
                z_0_1 = sample_z(mean_0_1, stddec_0_1)  # 期望与z_1一致
                x_0_1 = dec(z_0_1)  # 使用迁移编码经过解码器得到的迁移少数类样本

                # 少数类样本重构
                mean_and_stddec_1 = enc(x_1)  # 使用编码器对少数类样本进行编码
                mean_1, stddec_1 = tf.split(mean_and_stddec_1, 2, 1)  # 将1x128拆分为2x64，获取分布的两个参数
                z_1 = sample_z(mean_1, stddec_1)  # 根据分布的两个参数得到分布然后采样取值
                x_1_1 = dec(z_1)  # 由少数类样本重构得到的少数类样本

                # 原样本，重构样本以及迁移样本通过判别器
                dis_output_1 = dis(x=x_1)
                dis_output_1_1 = dis(x=x_1_1)
                dis_output_0_1 = dis(x=x_0_1)

                # 不同编码通过编码的判别器
                dis_map_output_1 = dis_map(x=z_1)
                dis_map_output_0_1 = dis_map(x=z_0_1)

                # VAE的损失（alpha_1*先验+alpha_2*似然）
                vae_0_0_likelihood_loss = get_vae_likelihood_loss(original_x=x_0, generated_x=x_0_0)
                vae_1_1_likelihood_loss = get_vae_likelihood_loss(original_x=x_1, generated_x=x_1_1)
                vae_0_0_loss = get_vae_loss(mean=mean_0, stddec=stddec_0, original_x=x_0, generated_x=x_0_0,
                                            alpha_1=alpha_1, alpha_2=alpha_2)
                vae_1_1_loss = get_vae_loss(mean=mean_1, stddec=stddec_1, original_x=x_1, generated_x=x_1_1,
                                            alpha_1=alpha_1, alpha_2=alpha_2)

                # 解码器生成迁移样本的损失
                dec_gen_migration_loss = - tf.math.log(tf.reduce_mean(dis_output_0_1))

                # 多数类迁移少数类编码与少数类编码的一致性损失   # 需要求一个方向上的均值
                code_consistency_loss = tf.reduce_mean(
                    tf.square(tf.reduce_mean(z_1, axis=0) - tf.reduce_mean(z_0_1, axis=0)))

                # 欧氏距离约束    # 需要求一个方向上的均值
                loss_distance_0_1_to_1 = tf.reduce_mean(
                    tf.square(tf.reduce_mean(x_0_1, axis=0) - tf.reduce_mean(x_1, axis=0)))
                loss_distance_0_1_to_0 = tf.reduce_mean(
                    tf.square(tf.reduce_mean(x_0_1, axis=0) - tf.reduce_mean(x_0, axis=0)))
                loss_distance = 2 * loss_distance_0_1_to_1 + 1 * loss_distance_0_1_to_0

                # 映射网络转换编码的损失
                loss_map_enc = - tf.math.log(tf.reduce_mean(dis_map_output_0_1))

                # 需要得到的损失
                enc_loss = vae_0_0_loss + vae_1_1_loss
                dec_loss = vae_0_0_likelihood_loss + vae_1_1_likelihood_loss + dec_gen_migration_loss + loss_distance
                map_net_loss = code_consistency_loss + loss_distance + loss_map_enc
                # dis_loss = 2.0 * tf.reduce_mean(dis_output_0_1) - tf.reduce_mean(dis_output_1_1) - tf.reduce_mean(dis_output_1)

                dis_loss = - (
                            tf.math.log(tf.reduce_mean(dis_output_1)) + tf.math.log(1 - tf.reduce_mean(dis_output_0_1)))
                dis_map_loss = - (tf.math.log(tf.reduce_mean(dis_map_output_1)) + tf.math.log(
                    1 - tf.reduce_mean(dis_map_output_0_1)))
                # 
                # loss = enc_loss + dec_gen_migration_loss + loss_distance + 1.2 * map_net_loss + dis_loss

                '''
                    # 将上述损失添加到loss_df中
                    # 需要把tensor数据类型转为float
                    # loss_df = loss_df.append([{'epoch':epoch, 
                    #                             'enc_loss':float(enc_loss.numpy()), 
                    #                             'dec_loss':float(dec_loss.numpy()), 
                    #                             'map_net_loss':float(map_net_loss.numpy()), 
                    #                             'dis_loss':float(dis_loss.numpy())}], ignore_index=True)
                '''
            # 计算梯度，优化编码器，解码器以及判别器
            grads = enc_tape.gradient(enc_loss, enc.trainable_variables)
            optimizer_enc.apply_gradients(zip(grads, enc.trainable_variables))

            grads = dec_tape.gradient(dec_loss, dec.trainable_variables)
            optimizer_dec.apply_gradients(zip(grads, dec.trainable_variables))

            grads = map_net_tape.gradient(map_net_loss, map_net.trainable_variables)
            optimizer_map_net.apply_gradients(zip(grads, map_net.trainable_variables))

            grads = dis_tape.gradient(dis_loss, dis.trainable_variables)
            optimizer_dis.apply_gradients(zip(grads, dis.trainable_variables))

            grads = dis_map_tape.gradient(dis_map_loss, dis_map.trainable_variables)
            optimizer_dis_map.apply_gradients(zip(grads, dis_map.trainable_variables))

        # 保存每一个epoch后的网络结构，并用测试验证集的性能指标
        optimized_smsg_net_dict = {'enc': enc, 'dec': dec, 'map_net': map_net, 'dis': dis}
        optimized_smsg_net_dict_list.append(optimized_smsg_net_dict)
        balanced_train_dataset_df = generate_new_1_sample(optimized_smsg_net_dict, train_batch, columns=columns)
        val_dataset_df = batch_list_to_df(val_batch, columns=columns)

        val_dataset_performance_sum = np.array(
            get_three_methods_performance_df(balanced_train_dataset_df, val_dataset_df).iloc[:, 1:]).astype(
            'double').sum()
        # print(val_dataset_performance_sum)
        val_performance_list.append(val_dataset_performance_sum)

        # 每100个epoch
        if (epoch + 1) % 100 == 0:
            # if True:
            #     # 每100个epoch绘制一下图像，检验有没有后验崩塌的现象存在
            #     optimized_smsg_net_dict = {'enc':enc, 'dec':dec, 'map_net':map_net, 'dis':dis}
            #     balanced_train_dataset_df = generate_new_1_sample(optimized_smsg_net_dict, train_batch, columns=columns)
            #     x1, y1 = balanced_train_dataset_df.iloc[:, :-1].astype('float'), balanced_train_dataset_df.iloc[:, -1].astype('int')
            #     x1 = pca.transform(x1)
            #     fig1 = sns.stripplot(x=x1[:, 0], y=x1[:, 1], hue=y1)
            #     scatter_fig1 = fig1.get_figure()
            #     scatter_fig1.savefig('/home/lqw/testone/my_code_pro_pro_pro/smsg/images/' + 'balanced_pima_' + str(index + 1) + '_' + str(epoch + 1) + '.png')
            #     fig1.clear()

            print('epoch' + str(epoch + 1) + '训练完成！', end='\t')

    # 将每个epoch训练得到的loss_df写入到文件中
    # loss_df.to_csv('smsg/loss_logger/train_loss.csv', index=False, header=True, sep=',')

    # 优化过的SMSG的网络结构
    # optimized_smsg_net_dict = {'enc':enc, 'dec':dec, 'map_net':map_net, 'dis':dis}
    # 使用300个epoch中对验证集效果最好的一组网络结构作为最终的网络结构
    max_index = val_performance_list.index(max(val_performance_list))
    print('最好的epoch是：' + str(max_index))
    optimized_smsg_net_dict = optimized_smsg_net_dict_list[max_index]

    return optimized_smsg_net_dict
