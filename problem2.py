import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from problem1 import read_excel


# 读取数据
def get_magnetic_density():
    data = pd.read_excel("excel/train1.xlsx", sheet_name="材料1")
    data = data[data['励磁波形'] == '正弦波']
    return data


# 数据准备
def get_dates(data):
    temperature = data.iloc[:, 0]  # 温度
    frequency = data.iloc[:, 1]  # 频率
    flux_density = data.iloc[:, 4:]  # 磁通密度
    flux_density_max = flux_density.max(axis=1).values # 磁通密度峰值
    loss = data.iloc[:, 2]  # 磁芯损耗
    return temperature, frequency, loss, flux_density_max


# 定义修正后的斯坦麦茨方程模型
def steinmetz_model(T, f, B, k0, k1, k2, a, b):
    k_T = k0 + k1 * T + k2 * T ** 2  # 温度相关的 k(T)
    return k_T * f ** a * B ** b


# 拟合函数
def fit_function(X, k0, k1, k2, a, b):
    T, f, B = X
    return steinmetz_model(T, f, B, k0, k1, k2, a, b)


# 最小二乘法拟合
def least_squares(data):
    temperature, frequency, loss, flux_density_max = get_dates(data)
    # 确保所有输入是一维数组
    print(temperature.shape, frequency.shape, flux_density_max.shape)
    X_data = np.vstack((temperature, frequency, flux_density_max))
    # 初始猜测参数
    initial_guess = [1e-6, 1e-8, 1e-10, 1.5, 2.0]
    # 使用 curve_fit 进行拟合
    params, covariance = curve_fit(fit_function, X_data, loss, p0=initial_guess)
    # 提取拟合参数
    k0, k1, k2, a, b = params
    print(f"拟合结果: k0 = {k0}, k1 = {k1}, k2 = {k2}, a = {a}, b = {b}")
    # 计算预测损耗
    predicted_loss = steinmetz_model(temperature, frequency, flux_density_max, k0, k1, k2, a, b)
    return predicted_loss


# 绘制损耗差距图
def plot_loss_difference(data):
    temperature, frequency, actual_loss, flux_density_max = get_dates(data)
    predicted_loss = least_squares(data)
    #计算决定系数
    r2 = 1 - np.sum((actual_loss - predicted_loss) ** 2) / np.sum((actual_loss - np.mean(actual_loss)) ** 2)
    # print(f"决定系数: {r2}")    0.995456411158246
    # # 绘制实际损耗与预测损耗的比较
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    # plt.rcParams['axes.unicode_minus'] = False
    # # 计算残差
    # residuals = actual_loss - predicted_loss
    # # 绘制残差图
    # plt.figure(figsize=(10, 6))
    # plt.scatter(np.arange(len(residuals)), residuals, label='残差')
    # plt.axhline(y=0, color='r', linestyle='--')  # 添加零线
    # plt.xlabel('样本点')
    # plt.ylabel(r'磁芯损耗差距（w/m$^3$）')
    # plt.title('实际磁芯损耗与预测磁芯损耗对比')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
# 幂函数拟合
if __name__ == '__main__':
    data = get_magnetic_density()
    plot_loss_difference(data)
