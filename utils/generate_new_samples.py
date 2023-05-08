"""
从优化好的网络中生成新的少数类样本
"""

import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from utils.data_process import batch_list_to_df
from utils.vae import sample_z


def generate_new_1_sample(optimized_smsg_net_dict, train_batch, columns):
    """
    从优化好的网络中生成新的少数类样本，将其添加到原来的训练集中
    :param : optimized_smsg_net_dict : 优化好的SMSG_PRO网络
    :param : train_batch : 训练集batches，包括多数类样本和少数类样本的batches
    :param : columns : df的列名

    :return : balanced_train_dataset_df : 经过平衡后的训练数据集batch
    """

    # 取出优化好的SMSG网络的编码器0和解码器1
    enc = optimized_smsg_net_dict.get('enc')
    dec = optimized_smsg_net_dict.get('dec')
    map_net = optimized_smsg_net_dict.get('map_net')

    # 初始化需要添加到训练集的迁移少数类batch_list
    migrate_data_1_batch_list = []

    # 取出每一个训练data_0_batch
    for data_0_batch in train_batch[0]:
        x_0, y_0 = data_0_batch
        # 类型转换，将x_0中数据类型转为tf.float32
        x_0 = tf.cast(x_0, dtype=tf.float32)
        mean_and_stddec_0 = enc(x_0)  # 使用优化好的编码器对多数类样本进行编码

        # 多数类样本迁移得到迁移少数类样本
        mean_and_stddec_0_1 = map_net(mean_and_stddec_0)  # 多数类样本的编码经过映射网络得到少数类样本的迁移编码
        mean_0_1, stddec_0_1 = tf.split(mean_and_stddec_0_1, 2, 1)  # 获取分布的两个参数
        z_0_1 = sample_z(mean_0_1, stddec_0_1)  # 期望与z_1一致
        x_0_1 = dec(z_0_1)  # 使用迁移编码经过解码器得到的迁移少数类样本
        y_0_1 = tf.constant(1.0, shape=y_0.shape)  # 对标签进行赋值1.0
        added_data_1_batch = (x_0_1, y_0_1)
        # 将每一次生成的样本batch添加到列表中
        migrate_data_1_batch_list.append(added_data_1_batch)

    # 得到原始多数类，原始少数类，迁移少数类的df
    data_0_df = batch_list_to_df(train_batch[0], columns=columns)
    data_1_df = batch_list_to_df(train_batch[1], columns=columns)

    migrate_data_1_df = batch_list_to_df(migrate_data_1_batch_list, columns=columns)
    # 进行一些深拷贝
    migrate_data_1_df2 = migrate_data_1_df.copy(deep=True)
    migrate_data_1_df3 = migrate_data_1_df.copy(deep=True)
    data_0_df2 = data_0_df.copy(deep=True)
    data_0_df3 = data_0_df.copy(deep=True)
    data_1_df2 = data_1_df.copy(deep=True)
    data_1_df3 = data_1_df.copy(deep=True)

    # data_0_df.to_csv('/home/lqw/testone/ttgan/ttgan/temp/data_0_df.csv', index=False, header=True, sep=',')
    # data_1_df.to_csv('/home/lqw/testone/ttgan/ttgan/temp/data_1_df.csv', index=False, header=True, sep=',')
    # migrate_data_1_df.to_csv('/home/lqw/testone/ttgan/ttgan/temp/migrate_data_1_df.csv', index=False, header=True, sep=',')

    # 从迁移生成的少数类样本的df中随机抽取原始多数类和原始少数类的差值个样本，保证生成后总的多数类和少数类样本数目相同
    # migrate_data_1_df = migrate_data_1_df.sample(n=data_0_df.shape[0]-data_1_df.shape[0], random_state=2022)

    migrate_data_1_df_by_lr = select_1_samples_by_lr(data_0_df, data_1_df, migrate_data_1_df)
    balanced_train_dataset_df_by_lr = pd.concat([data_0_df, data_1_df, migrate_data_1_df_by_lr], axis=0)

    migrate_data_1_df_by_rf = select_1_samples_by_rf(data_0_df2, data_1_df2, migrate_data_1_df2)
    balanced_train_dataset_df_by_rf = pd.concat([data_0_df, data_1_df, migrate_data_1_df_by_rf], axis=0)

    migrate_data_1_df_by_svm = select_1_samples_by_svm(data_0_df3, data_1_df3, migrate_data_1_df3)
    balanced_train_dataset_df_by_svm = pd.concat([data_0_df, data_1_df, migrate_data_1_df_by_svm], axis=0)

    return {'lr': balanced_train_dataset_df_by_lr, 'rf': balanced_train_dataset_df_by_rf,
            'svm': balanced_train_dataset_df_by_svm}


def select_1_samples_by_lr(data_0_df, data_1_df, migrate_data_1_df):
    original_data = pd.concat([data_0_df, data_1_df], axis=0)
    train_x = original_data.iloc[:, :-1]
    train_y = original_data.iloc[:, -1]
    test_x = migrate_data_1_df.iloc[:, :-1]

    lr = LogisticRegression()
    lr.fit(train_x, train_y)
    test_x = test_x.fillna(0)
    proba_array = lr.predict_proba(test_x)[:, -1]
    migrate_data_1_df['proba'] = proba_array
    migrate_data_1_df.sort_values(by="proba", inplace=True, ascending=True)
    migrate_data_1_df.index = range(len(migrate_data_1_df))
    m = data_0_df.shape[0] - data_1_df.shape[0]
    migrate_data_1_df = migrate_data_1_df.iloc[[i for i in range(m)], :]
    # 再将最后一列删除
    selected_migrate_data_1_df = migrate_data_1_df.drop(columns='proba')

    return selected_migrate_data_1_df


def select_1_samples_by_rf(data_0_df, data_1_df, migrate_data_1_df):
    original_data = pd.concat([data_0_df, data_1_df], axis=0)
    train_x = original_data.iloc[:, :-1]
    train_y = original_data.iloc[:, -1]
    test_x = migrate_data_1_df.iloc[:, :-1]

    rf = RandomForestClassifier()
    rf.fit(train_x, train_y)
    test_x = test_x.fillna(0)
    proba_array = rf.predict_proba(test_x)[:, -1]
    migrate_data_1_df['proba'] = proba_array
    migrate_data_1_df.sort_values(by="proba", inplace=True, ascending=True)
    migrate_data_1_df.index = range(len(migrate_data_1_df))
    m = data_0_df.shape[0] - data_1_df.shape[0]
    migrate_data_1_df = migrate_data_1_df.iloc[[i for i in range(m)], :]
    # 再将最后一列删除
    selected_migrate_data_1_df = migrate_data_1_df.drop(columns='proba')

    return selected_migrate_data_1_df


def select_1_samples_by_svm(data_0_df, data_1_df, migrate_data_1_df):
    original_data = pd.concat([data_0_df, data_1_df], axis=0)
    train_x = original_data.iloc[:, :-1]
    train_y = original_data.iloc[:, -1]
    test_x = migrate_data_1_df.iloc[:, :-1]

    svm = SVC(probability=True)
    svm.fit(train_x, train_y)
    test_x = test_x.fillna(0)
    proba_array = svm.predict_proba(test_x)[:, -1]
    migrate_data_1_df['proba'] = proba_array
    migrate_data_1_df.sort_values(by="proba", inplace=True, ascending=True)
    migrate_data_1_df.index = range(len(migrate_data_1_df))
    m = data_0_df.shape[0] - data_1_df.shape[0]
    migrate_data_1_df = migrate_data_1_df.iloc[[i for i in range(m)], :]
    # 再将最后一列删除
    selected_migrate_data_1_df = migrate_data_1_df.drop(columns='proba')

    return selected_migrate_data_1_df
