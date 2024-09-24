import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution
import xgboost as xgb

from logger import logger
from problem4 import flux_density_pca


# 加载问题4的磁芯损耗模型预测模型
def get_problem4_model():
    model = xgb.XGBRegressor()
    model_path = 'xgb_model.json'

    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' does not exist.")
        return None

    try:
        model.load_model(model_path)
        print(f"Model loaded successfully from '{model_path}'.")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None




# 目标函数
def objective_function(params, model):
    temperature, frequency, waveform_type, material, flux_density_max = params
    # 确保所有参数具有一致的维

    # 预测损耗
    input_data = np.array([temperature, frequency, waveform_type, material, flux_density_max]).reshape(1, -1)
    loss = model.predict(input_data)[0]
    # 传输磁能 = 频率 * 磁通密度峰值
    transmission_energy = frequency * flux_density_max
    # 综合目标函数：加权（这里简单相加，可依据需求调整权重）
    combined_objective = loss - transmission_energy  # 最大化 / 最小化
    return combined_objective


# 定义目标函数用于遗传算法
def optimized_objective(params, damage_model):
    return objective_function(params, damage_model)


def t5():
    damage_model = get_problem4_model()  # 损耗模型
    # 参数边界（根据实际的数据范围设置）
    bounds = [(25, 90),  # Temperature range
              (50000, 500000),  # Frequency range
              (1, 3),  # Waveform type: 1=正弦波, 2=三角波, 3=梯形波
              (1, 4),  # Material type: 1, 2, 3, 4
              (-0.312418839, 0.313284469)]  # Magnetic Flux Density Peak range 0.313284469, -0.312418839

    np.random.seed(42)
    # 初始化参数
    initial_params = [50, 100000, 1, 1, 0.1]
    # 定义目标函数用于遗传算法
    optimized_objective(initial_params, damage_model)
    # 使用差分进化算法 Differential Evolution
    logger.info("Starting optimization...")
    result = differential_evolution(lambda x: optimized_objective(x, damage_model), bounds,seed=42)
    print(result)
    # plot_result(result)  # 绘画图
    logger.info("Optimization completed.")
    # 输出最优参数
    optimal_params = result.x
    optimal_loss = damage_model.predict([optimal_params])[0]
    optimal_transmission_energy = optimal_params[1] * optimal_params[4]

    logger.info("Optimal parameters:")
    logger.info(f"Temperature: {optimal_params[0]}°C")
    logger.info(f"Frequency: {optimal_params[1]} Hz")
    logger.info(f"Waveform type: {int(optimal_params[2])}")
    logger.info(f"Material type: {int(optimal_params[3])}")
    logger.info(f"Magnetic Flux Density Peak: {optimal_params[4]} T")
    logger.info(f"Optimal Loss: {optimal_loss}")
    logger.info(f"Optimal Transmission Energy: {optimal_transmission_energy}")
    logger.info("Ending problem5")


def compute_Magnetic_Flux_Density_Peak_range():
    data=pd.read_excel("excel/merge_train2.xlsx")
    Magnetic_Flux_Density=data.iloc[:, 5:]
    max=Magnetic_Flux_Density.max().max()
    min=Magnetic_Flux_Density.min().min()
    return max, min

def plot_result(result):
    # 绘制适应度值变化图
    # 提取所需数据
    fitness_history = result.get('all_fitness', [])  # 适应度历史记录
    iterations = list(range(len(fitness_history)))  # 迭代次数

    # 绘制适应度值变化图
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, fitness_history, label='Best Fitness Value')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.title('Convergence of Differential Evolution')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    logger.info("Starting problem5")
    # max,min=compute_Magnetic_Flux_Density_Peak_range()
    t5()
    logger.info("Ending problem5")