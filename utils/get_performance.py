"""
获取在三种方法上的性能表现f1和gmean
"""

import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score
from sklearn.svm import SVC


def get_svm_performance_dict(train_datatset_df, test_datatset_df):
    """
    获取平衡后的数据集在SVM上的表现
    :param : train_datatset_df : 平衡后的训练集
    :param : test_datatset_df : 原测试集

    :return : {'method_name': 'svm', 'f1':my_f1, 'gmean':my_gmean}
    """
    
    train_x = np.array(train_datatset_df.iloc[:, :-1])
    train_y = np.array(train_datatset_df.iloc[:, -1])
    test_x = np.array(test_datatset_df.iloc[:, :-1])
    test_y = np.array(test_datatset_df.iloc[:, -1])
    # 初始化SVM
    svm = SVC()
    svm.fit(train_x, train_y)
    predicted_y = svm.predict(test_x)
    # 获取性能
    my_f1 = f1_score(test_y, predicted_y, average=None)[1]
    recall = recall_score(test_y, predicted_y, average=None)
    my_gmean = math.sqrt(recall[0] * recall[1])
    return {'method_name': 'svm', 'f1':my_f1, 'gmean':my_gmean}


def get_lr_performance_dict(train_datatset_df, test_datatset_df):
    """
    获取平衡后的数据集在LR上的表现
    :param : train_datatset_df : 平衡后的训练集
    :param : test_datatset_df : 原测试集

    :return : {'method_name': 'lr', 'f1':my_f1, 'gmean':my_gmean}
    """

    
    train_x = np.array(train_datatset_df.iloc[:, :-1])
    train_y = np.array(train_datatset_df.iloc[:, -1])
    test_x = np.array(test_datatset_df.iloc[:, :-1])
    test_y = np.array(test_datatset_df.iloc[:, -1])

    # 初始化lr
    lr = LogisticRegression()
    lr.fit(train_x, train_y)
    predicted_y = lr.predict(test_x)
    # 获取性能
    my_f1 = f1_score(test_y, predicted_y, average=None)[1]  # index=0得到的是类0的F1，index=1得到的是类1的F1，这里我们需要的是类1（少数类）的F1
    recall = recall_score(test_y, predicted_y, average=None)
    my_gmean = math.sqrt(recall[0] * recall[1])
    return {'method_name': 'lr', 'f1':my_f1, 'gmean':my_gmean}


def get_rf_performance_dict(train_datatset_df, test_datatset_df):
    """
    获取平衡后的数据集在RF上的表现
    :param : train_datatset_df : 平衡后的训练集
    :param : test_datatset_df : 原测试集

    :return : {'method_name': 'rf', 'f1':my_f1, 'gmean':my_gmean}
    """
    
    train_x = np.array(train_datatset_df.iloc[:, :-1])
    train_y = np.array(train_datatset_df.iloc[:, -1])
    test_x = np.array(test_datatset_df.iloc[:, :-1])
    test_y = np.array(test_datatset_df.iloc[:, -1])

    # 初始化lr
    rf = RandomForestClassifier()
    rf.fit(train_x, train_y)
    predicted_y = rf.predict(test_x)
    # 获取性能
    my_f1 = f1_score(test_y, predicted_y, average=None)[1]
    recall = recall_score(test_y, predicted_y, average=None)
    my_gmean = math.sqrt(recall[0] * recall[1])
    return {'method_name': 'rf', 'f1':my_f1, 'gmean':my_gmean}


def get_three_methods_performance_df(train_datatset_df_dict, test_datatset_df):
    """
    综合上述三个方法，将上述三个方法得到的字典合成为一个pandas.DataFrame
    :param : train_datatset_df_dict : 平衡后的训练集字典
    :param : test_datatset_df : 原测试集
    
    :return : three_methods_performance_df
               method       f1          gmean
                svm      f1-value     gmean-value
                lr       f1-value     gmean-value
                rf       f1-value     gmean-value
    """
    train_datatset_df_lr = train_datatset_df_dict.get('lr')
    train_datatset_df_rf = train_datatset_df_dict.get('rf')
    train_datatset_df_svm = train_datatset_df_dict.get('svm')

    svm_performance_dict = get_svm_performance_dict(train_datatset_df_svm, test_datatset_df)
    lr_performance_dict = get_lr_performance_dict(train_datatset_df_lr, test_datatset_df)
    rf_performance_dict = get_rf_performance_dict(train_datatset_df_rf, test_datatset_df)

    method_list = [svm_performance_dict.get('method_name'), lr_performance_dict.get('method_name'), rf_performance_dict.get('method_name')]
    lr_list = [svm_performance_dict.get('f1'), lr_performance_dict.get('f1'), rf_performance_dict.get('f1')]
    rf_list = [svm_performance_dict.get('gmean'), lr_performance_dict.get('gmean'), rf_performance_dict.get('gmean')]
    
    three_methods_performance_df = pd.DataFrame(data=np.transpose([method_list, lr_list, rf_list]), columns=['method', 'f1', 'gmean'])

    return three_methods_performance_df

