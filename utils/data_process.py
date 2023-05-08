"""
数据处理
    加载keel数据集数据
    数据预处理
    根据交叉验证获取训练集和测试机batch
    将一个batch转为array, df
    将一个batch_list转为array, df
"""


import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


def load_keel_dataset(dataset_path, keel_dataset_name):
    """
    加载keel数据集
    先获取keel数据集的特征和具体数据，依据此来构造df

    :param dataset_path : 数据集路径
    :param keel_dataset_name : 需要加载的keel数据集名称
    :return df : 返回加载的数据集的pandas.DataFrame数据格式
    """

    df = pd.read_csv(dataset_path + keel_dataset_name)
    # 字符串转为数字，忽略错误（默认返回dtype为float64或int64，具体取决于提供的数据。）
    df = df.apply(pd.to_numeric, errors='ignore')

    return df


def keel_dataset_preprocess(keel_dataset_df):
    """
    对pandas.DataFrame格式的keel_dataset进行预处理
    主要包括：
            对样本标签的操作：将多数类样本的标签赋值为0，少数类样本的标签赋值为1
            对样本属性的操作：进行特征提取，将非数字属性值转换为one-hot编码格式
                           将数据归一化到(-1, 1)范围内
    
    :param : keel_dataset_df : keel数据集经过数据集加载后得到的pandas.DataFrame格式

    :return : keel_dataset_df : 经过上述处理后的keel_dataset_df
    """
    # 添加一列权重的部分， 权重=10个近邻中   1.0 * (多数类样本的个数 + 1) / (少数类样本的个数 + 1)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(np.array(keel_dataset_df.iloc[:, :-1]), np.array(keel_dataset_df.iloc[:, -1]))
    label_sum_list = []
    for i in range(keel_dataset_df.shape[0]):
        knn_index = knn.kneighbors([(keel_dataset_df.iloc[i, :-1])], return_distance=False)
        label_sum = keel_dataset_df.iloc[knn_index[0], -1].sum()
        label_sum_list.append(1.0 * (10 - label_sum + 1) / (label_sum + 1))
    keel_dataset_df.insert(loc=len(keel_dataset_df.columns) - 1, column='knn', value=label_sum_list)
    
    return keel_dataset_df


def get_keel_dataset_batch_list(keel_dataset_df, batch_size=16, my_random_state=2022):
    """
    根据n折交叉验证和batch_size大小 获取经过预处理后keel数据集的batch_list

    :param : keel_dataset_df : 经过预处理后的keel数据集，数据类型为pandas.DataFrame
    :param : my_n_splits : 交叉验证参数，要分割为多少个子集
    :param : batch_size : batch大小
    :param : my_random_state : random_state大小

    :return : batch_list : [train_batch_list, val_batch_list]
    """
    # 初始化train_batch_list, val_batch_list
    train_batch_list, val_batch_list = [], []

    # 获取多数类样本和少数类样本的df并进行打乱
    data_0_df = keel_dataset_df[keel_dataset_df[keel_dataset_df.columns[-1]] == 0]
    data_1_df = keel_dataset_df[keel_dataset_df[keel_dataset_df.columns[-1]] == 1]
    data_0_df.index = range(data_0_df.shape[0])
    data_1_df.index = range(data_1_df.shape[0])
        
    class_0_train_index = list(data_0_df.index)
    class_1_train_index = list(data_1_df.index)

    class_0_val_index = random.sample(class_0_train_index, (int)(data_0_df.shape[0] / 4))
    class_1_val_index = random.sample(class_1_train_index, (int)(data_1_df.shape[0] / 4))

    class_0_train_index = list(set(class_0_train_index).difference(class_0_val_index))
    class_1_train_index = list(set(class_1_train_index).difference(class_1_val_index))

    # 根据上述index列表，获取一次交叉验证中所有的训练集样本
    train_dataset_0_df = data_0_df.iloc[class_0_train_index]
    train_dataset_1_df = data_1_df.iloc[class_1_train_index]

    # 根据上述index列表，获取一次交叉验证中所有的验证集样本
    val_dataset_0_df = data_0_df.iloc[class_0_val_index]
    val_dataset_1_df = data_1_df.iloc[class_1_val_index]

    train_dataset_df = pd.concat([train_dataset_0_df, train_dataset_1_df], axis=0)
    val_dataset_df = pd.concat([val_dataset_0_df, val_dataset_1_df], axis=0)
    
    # 获取 训练集中 多数类样本与少数类样本的比例 rate = 多数类样本数目 / 少数类样本数目
    rate = (len(class_0_train_index) // len(class_1_train_index)) + 1
    
    # 对batch_size大小进行调整，如果 训练集中少数类样本的数目 大于 原batch_size 的四倍，则不变，否则调整后的batch_size是训练集中少数类样本的数目大小的1/4
    if batch_size * 4 > len(class_1_train_index):
        batch_size = (len(class_1_train_index) // 4) + 1
    
    # 多数类样本和少数类类样本使用不同的batch_size，多数类类样本的batch_size = 调整后的batch_size（少数类类样本的batch_size）* rate
    train_batch_list = [tf.data.Dataset.from_tensor_slices((train_dataset_0_df.iloc[:, :-1].values, train_dataset_0_df.iloc[:, -1].values)).batch(batch_size*rate).shuffle(100), 
                            tf.data.Dataset.from_tensor_slices((train_dataset_1_df.iloc[:, :-1].values, train_dataset_1_df.iloc[:, -1].values)).batch(batch_size).shuffle(100)]
    
    
    val_batch_list = tf.data.Dataset.from_tensor_slices((val_dataset_df.iloc[:, :-1].values, val_dataset_df.iloc[:, -1].values)).batch(batch_size).shuffle(100)

    batch_list = [train_batch_list, val_batch_list]
    
    return batch_list


def batch_to_array(one_batch):
    """
    将一个batch转为np.array
    """
    # 分别获取属性值与标签值的array
    attribute_value_array = np.array(one_batch[0])
    label_array = np.transpose(np.array(one_batch[1]))
    label_array = label_array[:, np.newaxis]  # 升维  (n, ) ----> (n, 1)
    # 合并np.array
    one_batch_array = np.concatenate([attribute_value_array, label_array], axis=1)
        
    return one_batch_array


def batch_list_to_array(one_batch_list):
    """
    将 [batch_1, batch_2, ..., batch_n, ...] 转为array
    """
    one_batch_array_list = []
    # 获取每一个batch的array
    for one_batch in one_batch_list:
        one_batch_array = batch_to_array(one_batch=one_batch)
        one_batch_array_list.append(one_batch_array)
    # 纵向合并
    one_batch_list_array = np.concatenate(one_batch_array_list, axis=0)
    return one_batch_list_array


def batch_to_df(one_batch, columns):
    """
    将一个batch转为df
    """
    # 先将batch转为np.array
    one_batch_array = batch_to_array(one_batch=one_batch)
    # 将np.array转为df
    df = pd.DataFrame(one_batch_array, columns=columns)

    return df


def batch_list_to_df(one_batch_list, columns):
    """
    将 [batch_1, batch_2, ..., batch_n, ...] 转为df
    """

    # 将one_batch_list转为np.array
    one_batch_list_array = batch_list_to_array(one_batch_list=one_batch_list)

    # 将np.array转为df
    df = pd.DataFrame(one_batch_list_array, columns=columns) 

    return df



