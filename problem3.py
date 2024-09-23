import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

from logger import logger

'''
第三题:
请根据附件一中的实验数据，通过数据分析技术，分析温度、励磁波形和磁芯材料这三个因素，
是如何独立及协同影响着磁芯损耗（仅讨论两两之间协同影响）；
以及他们各自的影响程度；并给出这三个因素在什么条件下，磁芯损耗可能达到最小？
'''


# 读取不同材料合并后的数据
def read_merged_data():
    data = pd.read_excel('excel/merge_train2.xlsx')
    return data


def get_merged_data(data):
    temperature = data['温度'].values  # 温度
    frequency = data['频率'].values  # 频率
    waveform_type = data['励磁波形'].values  # 励磁波形
    flux_density = data.iloc[:, 5:]  # 磁通密度
    material = data['材料'].values  # 材料
    # flux_density_max = flux_density.max(axis=1).values  # 磁通密度峰值
    loss = data['磁芯损耗'].values  # 磁芯损耗
    return temperature, frequency, loss, waveform_type, flux_density, material


# 温度对磁芯损耗的正态性检验，计算p-value
def compute_temperature_loss_normality(data):
    temperature, frequency, loss, waveform_type, flux_density, material = get_merged_data(
        data)
    # logger.info(sm.stats.diagnostic.lilliefors(temperature, dist='norm'))
    # logger.info(sm.stats.diagnostic.lilliefors(loss, dist='norm'))
    # logger.info(sm.stats.diagnostic.lilliefors(waveform_type, dist='norm'))
    # 控制温度和波形相同，进行材料和磁芯损耗的斯皮尔慢相关性分析
    grouped_data = data.groupby(['温度', '励磁波形'])

    plot_3d3(data)
    # spearmanr_corr, p_value = stats.spearmanr(group['材料'], group['磁芯损耗'])
    # logger.info(
    #     f"For 温度={temp} and 波形={mat}, sPearson Correlation between Material and Loss is {spearmanr_corr} (p-value: {p_value})")
    # 控制材料和温度相同
    # grouped_data = data.groupby(['温度', '材料'])
    # for name, group in grouped_data:
    #     temp, mat = name
    #     spearmanr_corr, p_value = stats.spearmanr(group['励磁波形'], group['磁芯损耗'])
    #     logger.info(
    #         f"For 温度={temp} and 材料={mat}, sPearson Correlation between Material and Loss is {spearmanr_corr} (p-value: {p_value})")
    # # 控制波形和材料
    # grouped_data = data.groupby(['励磁波形', '材料'])
    # for name, group in grouped_data:
    #     temp, mat = name
    #     spearmanr_corr, p_value = stats.spearmanr(group['温度'], group['磁芯损耗'])
    #     logger.info(
    # f"For 波形={temp} and 材料={mat}, sPearson Correlation between Material and
    # Loss is {spearmanr_corr} (p-value: {p_value})")


def plot_3d(data):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    plt.rcParams['axes.unicode_minus'] = False
    # 假设 `data` 已经定义并且包含了必要的列
    # 获取唯一的温度和励磁波形值
    mat_unique = data['励磁波形'].unique()
    # 创建2D图
    plt.figure(figsize=(10, 6))
    # 定义励磁波形的对应关系
    waveform_labels = {1: '正弦波', 2: '三角波', 3: '梯形波'}
    # 循环遍历每个励磁波形值
    for mat in mat_unique:
        # 提取当前励磁波形下的数据
        group = data[data['励磁波形'] == mat]
        # 计算每个温度下的磁芯损耗平均值
        avg_core_loss = group.groupby('温度')['磁芯损耗'].mean()
        # 绘制折线图并加上标记点
        plt.plot(
            avg_core_loss.index,
            avg_core_loss.values,
            marker='o',
            label=f'{waveform_labels[mat]}')
    # 设置坐标轴标签和图例
    plt.xlabel('温度')
    plt.ylabel('磁芯损耗（平均值）')
    plt.title('不同励磁波形下的温度与磁芯损耗关系')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_3d2(data):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    plt.rcParams['axes.unicode_minus'] = False
    # 假设 `data` 已经定义并且包含了必要的列
    # 获取唯一的温度和励磁波形值
    mat_unique = data['材料'].unique()
    plt.figure(figsize=(10, 6))
    # 定义励磁波形的对应关系
    waveform_labels = {1: '正弦波', 2: '三角波', 3: '梯形波'}
    mat_labels = {1: '材料1', 2: '材料2', 3: '材料3', 4: '材料4'}
    # 循环遍历每个励磁波形值
    for mat in mat_unique:
        # 提取当前励磁波形下的数据
        group = data[data['材料'] == mat]
        # 计算每个温度下的磁芯损耗平均值
        avg_core_loss = group.groupby('温度')['磁芯损耗'].mean()
        # 绘制折线图并加上标记点
        plt.plot(
            avg_core_loss.index,
            avg_core_loss.values,
            marker='o',
            label=f'{mat_labels[mat]}')
    # 设置坐标轴标签和图例
    plt.xlabel('温度')
    plt.ylabel('磁芯损耗（平均值）')
    plt.title('不同材料下的温度与磁芯损耗关系')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_3d3(data):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    plt.rcParams['axes.unicode_minus'] = False
    # 假设 `data` 已经定义并且包含了必要的列
    # 获取唯一的温度和励磁波形值
    mat_unique = data['励磁波形'].unique()
    # 创建2D图
    plt.figure(figsize=(10, 6))
    # 定义励磁波形的对应关系
    waveform_labels = {1: '正弦波', 2: '三角波', 3: '梯形波'}
    mat_labels = {1: '材料1', 2: '材料2', 3: '材料3', 4: '材料4'}

    # 循环遍历每个励磁波形值
    for mat in mat_unique:
        # 提取当前励磁波形下的数据
        group = data[data['励磁波形'] == mat]
        # 计算每个材料下的磁芯损耗平均值
        avg_core_loss = group.groupby('材料')['磁芯损耗'].mean()
        # 绘制折线图并加上标记点
        plt.plot(avg_core_loss.index.map(lambda x: mat_labels[x]), avg_core_loss.values, marker='o',
                 label=f'{waveform_labels[mat]}')
    # 设置坐标轴标签和图例
    plt.xlabel('材料')
    plt.ylabel('磁芯损耗（平均值）')
    plt.title('不同波形下的材料与磁芯损耗关系')
    plt.legend()
    plt.grid(True)
    plt.show()


# 协同模型定义
def steinmetz_model(B, material, temperature, waveform, frequency):
    loss = B[0] + B[1] * material + B[2] * temperature + B[3] * waveform + B[4] * frequency + B[
        5] * material * temperature + B[6] * material * waveform + +B[7] * temperature * waveform + B[
        8] * material * temperature * waveform

def get_result_problem(data):
    # 当温度为90，材料为材料4，波形为正形波，计算最低的磁芯损耗
    result = data[(data['温度'] == 90) & (data['材料'] == 4) & (data['励磁波形'] == 1)]
    return result['磁芯损耗'].min()
#
def t3():
    data = read_merged_data()
    # compute_temperature_loss_normality(data)
    logger.info(get_result_problem(data))


if __name__ == '__main__':
    t3()
