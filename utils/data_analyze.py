"""
对得到的不同数据集的性能表现进行处理分析
    对每一个数据集得到的performance.csv文件进行处理，求不同方法的mean, max，得到处理后的performance_df
    对包含所有数据集的性能表现的文件夹进行处理，并将不同数据集的处理结果写入到目标文件夹中
    得到三种方法在不同数据集中的性能指标并将其保存在文件中

其他处理得到的性能数据的方法
"""


import os

import pandas as pd


def performance_csv_process(performance_csv_path):
    """
    对每一个数据集得到的performance.csv文件进行处理，求不同方法的mean, max，得到处理后的performance_df

    :param : performance_csv_path : csv文件路径
    :return : performance_df : columns=['method', 'mean_f1', 'mean_gmean', 'max_f1', 'max_gmean']
    """
    df = pd.read_csv(performance_csv_path)
    # 取出三种方法的五次交叉验证的性能指标
    df_svm = df.loc[df['method'] == 'svm']
    df_lr = df.loc[df['method'] == 'lr']
    df_rf = df.loc[df['method'] == 'rf']
    # 按列取均值
    svm_mean, svm_max = df_svm.mean(axis=0, numeric_only=True), df_svm.max(axis=0)
    lr_mean, lr_max = df_lr.mean(axis=0, numeric_only=True), df_lr.max(axis=0)
    rf_mean, rf_max = df_rf.mean(axis=0, numeric_only=True), df_rf.max(axis=0)
    # 归整一下
    svm = [svm_max[0], svm_mean[0], svm_mean[1], svm_max[1], svm_max[2]]
    lr = [lr_max[0], lr_mean[0], lr_mean[1], lr_max[1], lr_max[2]]
    rf = [rf_max[0], rf_mean[0], rf_mean[1], rf_max[1], rf_max[2]]
    performance_df = pd.DataFrame([lr, rf, svm], columns=['method', 'mean_f1', 'mean_gmean', 'max_f1', 'max_gmean'])
    
    return performance_df


def performance_folder_process(performance_folder_path, save_folder_path):
    """
    对包含所有数据集的性能表现的文件夹进行处理，并将不同数据集的处理结果写入到目标文件夹中
    :param : performance_folder_path : 不同数据集性能指标的文件夹路径
    :param : save_folder_path : 保存的文件夹路径
    """
    # 获取所有的性能指标df的name_list
    peformance_df_name_list = os.listdir(performance_folder_path)
    for performance_df_name in peformance_df_name_list:
        performance_df_after_process = performance_csv_process(performance_csv_path=performance_folder_path + performance_df_name)
        performance_df_after_process.to_csv(save_folder_path + performance_df_name, index=False, header=True, sep=',')


def get_synthetical_performance_df(performance_folder_path, save_folder_path):
    """
    获取三种方法在不同数据集上的综合df，并将处理结果写入到目标文件夹中
    :param : performance_folder_path : 不同数据集性能指标的文件夹路径
    :param : save_folder_path : 保存的文件夹路径
    """
    # 初始化df_list
    all_df_list = []
    
    # 获取所有的性能指标df的name_list
    peformance_df_name_list = os.listdir(performance_folder_path)
    for performance_df_name in peformance_df_name_list:
        dataset_name = performance_df_name[:-4]
        performance_df_after_process = performance_csv_process(performance_csv_path=performance_folder_path + performance_df_name)
        # 插入方法列
        performance_df_after_process.insert(loc=1, column='dataset_name', value=[dataset_name for i in range(3)])
        # 添加到列表中
        all_df_list.append(performance_df_after_process)
    
    # 纵向合并all_df_list
    all_df = pd.concat(all_df_list, axis=0)

    # 按照不同方法取出
    svm_df = all_df.loc[all_df['method']=='svm']
    lr_df = all_df.loc[all_df['method']=='lr']
    rf_df = all_df.loc[all_df['method']=='rf']

    # 保存
    svm_df.to_csv(save_folder_path + 'svm.csv', index=False, header=True, sep=',')
    lr_df.to_csv(save_folder_path + 'lr.csv', index=False, header=True, sep=',')
    rf_df.to_csv(save_folder_path + 'rf.csv', index=False, header=True, sep=',')


