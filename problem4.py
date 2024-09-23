'''
第四题：
1.首先获取特征
    磁芯密度的特征同第一题获取方法相同，利用时域和频域相结合，标准化、pca降维
    合并温度、材料、波形、频率、磁芯密度特征(对分类变量要进行转换)
    利用机器学习模型进行训练
    超参数调整
    预测附件三的数据
    填写数据
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mkl_fft import fft
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from skopt import BayesSearchCV
from tpot import TPOTRegressor
import xgboost as xgb
from logger import logger

'''获取磁性密度的特征'''


def extract_time_domain_features(data_flux_density):
    features = {}
    # 计算时域特征
    features['mean'] = np.mean(data_flux_density, axis=1)
    features['std'] = np.std(data_flux_density, axis=1)
    features['peak'] = np.max(data_flux_density, axis=1)
    features['peak_to_peak'] = np.ptp(data_flux_density, axis=1)
    features['skewness'] = skew(data_flux_density, axis=1)
    features['kurtosis'] = kurtosis(data_flux_density, axis=1)
    features['waveform_factor'] = np.abs(
        np.diff(np.unwrap(np.angle(fft(data_flux_density))), axis=1)).mean(axis=1)  # 确保输出是一维 波形因子
    features['rms'] = np.sqrt(np.mean(data_flux_density ** 2, axis=1))  # 均方根
    return pd.DataFrame(features)


# 计算频域特征
def extract_frequency_domain_features(data_flux_density):
    features = {}
    # 对每行数据进行FFT
    fft_data = fft(data_flux_density)
    # 提取频域特征（主频、能量等）
    features['fft_0'] = np.real(fft_data[:, 0])
    features['fft_1'] = np.real(fft_data[:, 1])
    features['fft_2'] = np.real(fft_data[:, 2])
    features['dominant_frequency'] = np.argmax(np.abs(fft_data), axis=1)
    features['spectral_energy'] = np.sum(np.abs(fft_data) ** 2, axis=1)
    return pd.DataFrame(features)


# 时域和频域特征合并,这里从5切割是因为前面添加了材料类别标记列
def merge_time_and_frequency_features(data):
    freq_features = extract_time_domain_features(data.iloc[:, 5:].values)
    time_features = extract_frequency_domain_features(data.iloc[:, 5:].values)
    return pd.concat([time_features, freq_features], axis=1)


# pca降维
def flux_density_pca(flux_density_features):
    # plot_pca(flux_density_features)      #绘制图表查看降维数
    scaler = StandardScaler()
    flux_density_feature = scaler.fit_transform(flux_density_features)
    pca = PCA(n_components=5)
    flux_density_pca = pca.fit_transform(flux_density_feature)
    return flux_density_pca


# 合并磁芯密度特征和温度、材料、波形、频率、磁芯密度特征
def get_merged_data(data, flux_density_pca):
    # 处理材料、波形分类变量进行数值化处理
    data['励磁波形'] = data['励磁波形'].astype('category').cat.codes
    data['材料'] = data['材料'].astype('category').cat.codes
    merged_data = pd.concat(
        [data.iloc[:, :5], pd.DataFrame(flux_density_pca)], axis=1)
    return merged_data


def plot_pca(flux_density_features):
    pca = PCA().fit(flux_density_features)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('累计解释方差比随主成分数量增加的变化曲线')
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('主成分数量')
    plt.ylabel('累计解释方差比')
    plt.show()


# 采用自动机器学习:弃用


def model_autoMl(merged_data):
    # 分离数据
    X = merged_data.drop('磁芯损耗', axis=1)
    y = merged_data['磁芯损耗']
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)
    # 创建AutoML模型
    tpot = TPOTRegressor(
        generations=10,
        population_size=50,
        verbosity=2,
        random_state=42)
    # 训练模型
    tpot.fit(X_train, y_train)
    # 预测并评估模型
    y_pred = tpot.predict(X_test)
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
    # 输出最佳模型的Python代码
    tpot.export('best_model_pipeline.py')


# 采用XGBoost


def model(merged_data, test_data):
    # 分离数据
    X = merged_data.drop('磁芯损耗', axis=1)
    y = merged_data['磁芯损耗']
    # 划分数据集:训练集：验证集，8:2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)
    # 初始化XGBoost回归模型
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=8,
        random_state=42)
    # 初始化LGMOdel回归模型
    # xgb_model = lgb.LGBMRegressor(
    #     n_estimators=100,
    #     learning_rate=0.1,
    #     max_depth=8,
    #     random_state=42)
    # 进行5折交叉验证
    cv_scores = cross_val_score(
        xgb_model,
        X_train,
        y_train,
        cv=5,
        scoring='neg_mean_squared_error')
    cv_mse = -cv_scores.mean()
    print(f'交叉验证均方误差: {cv_mse}')
    xgb_model.fit(X_train, y_train)
    # 预测结果
    y_pred_xgb = xgb_model.predict(X_test)
    # 评估模型
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    print(f'XGBoost模型的均方误差: {mse_xgb}')
    # 计算R的平方
    r2_xgb = xgb_model.score(X_test, y_test)
    # 绘制图表
    plot_xgp(y_test, y_pred_xgb, r2_xgb)
    print(f'XGBoost模型的R的平方: {r2_xgb}')
    y_pred_test = xgb_model.predict(test_data)
    logger.info(y_pred_test)
    # 将预测值写入一个新的文件的磁芯损耗列中
    # test_data['预测值'] = y_pred_test
    # test_data.to_excel('test_data3.xlsx', index=False)


def model2(merged_data, test_data):
    # 分离数据
    X = merged_data.drop('磁芯损耗', axis=1)
    y = merged_data['磁芯损耗']

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 定义超参数分布
    param_space = {
        'n_estimators': (50, 200),
        'learning_rate': (0.01, 0.2),
        'max_depth': (3, 8),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0)
    }
    # 初始化XGBoost回归模型
    xgb_model = xgb.XGBRegressor(random_state=42)
    # 创建 BayesSearchCV 对象
    bayes_search = BayesSearchCV(xgb_model, param_space, n_iter=30, cv=5,
                                 scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    # 拟合模型，进行超参数优化
    bayes_search.fit(X_train, y_train)
    # 输出最佳参数
    print("Best parameters found: ", bayes_search.best_params_)
    print("Best score found: ", -bayes_search.best_score_)
    # 获取最佳模型
    best_model = bayes_search.best_estimator_
    # 使用最佳模型进行预测
    y_pred_xgb = best_model.predict(X_test)
    # 绘制图表
    plot_xgp(y_test, y_pred_xgb)
    # 评估模型
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    print(f'XGBoost模型的均方误差: {mse_xgb}')
    # 计算 R 方
    r2_xgb = best_model.score(X_test, y_test)
    print(f'XGBoost模型的R方: {r2_xgb}')
    # 使用测试集进行预测
    y_pred_test = best_model.predict(test_data)
    logger.info(y_pred_test)
    # 将预测值写入新的文件
    test_data['预测值'] = y_pred_test
    test_data.to_excel('test_data3.xlsx', index=False)
# 绘制能体现预测值和真实值的散点密度图


def plot_xgp(y_test, y_pred, r2_xgb):  # 创建散点密度图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    # 添加对角线（理想预测线）
    vmin = min(min(y_test), min(y_pred))
    vmax = max(max(y_test), max(y_pred))
    plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=2, label='理想值')
    # 图上添加R方:r2_xgb
    plt.text(0.2, 0.8, r'$R^2$: {:.4f}'.format(r2_xgb), transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', color='red')
    # 设置坐标轴标签和标题
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('真实值和预测值的关系')
    plt.legend()
    plt.grid(True)
    plt.show()


# 测试集数据处理
def tes_data_process(data):
    # 删除序号列
    data = data.drop('序号', axis=1)
    # 交换励磁波形和材料两列
    # 获取列名列表
    columns = data.columns.tolist()
    # 找到励磁波形和材料两列的位置
    try:
        index_excitation_waveform = columns.index('励磁波形')
        index_material = columns.index('材料')
        index_frequency = columns.index('频率')
        # 交换两列的位置
        columns[index_excitation_waveform], columns[index_material] = columns[index_material], columns[
            index_excitation_waveform]
        # 重新设置列顺序
        data = data[columns]
    # 在频率后面添加一空列,列头为励磁损耗，以便适用于之前训练集列头不一致
        data.insert(index_frequency + 1, '磁芯损耗', np.nan)
    except ValueError:
        print("未找到 '励磁波形' 或 '材料' 列，请检查数据列名。")
    # print(data.columns)
    data['励磁波形'] = data['励磁波形'].map({'正弦波': 1, '三角波': 2, '梯形波': 3})
    data['材料'] = data['材料'].map({'材料1': 1, '材料2': 2, '材料3': 3, '材料4': 4})
    flux_density_features = merge_time_and_frequency_features(
        data)
    flux_density = flux_density_pca(flux_density_features)
    merged_data = get_merged_data(data, flux_density)
    # 删除磁芯损耗列
    merged_data = merged_data.drop('磁芯损耗', axis=1)
    return merged_data


def t4():
    train_data = pd.read_excel("excel/merge_train2.xlsx")
    test_data = pd.read_excel("excel/test3.xlsx")
    test_merged_data = tes_data_process(test_data)
    flux_density_features = merge_time_and_frequency_features(
        train_data)  # 获取磁芯密度全部特征
    # 磁芯密度全部特征pca降维
    flux_density = flux_density_pca(flux_density_features)
    train_merged_data = get_merged_data(train_data, flux_density)  # 处理完的特征
    # 预测模型
    model(train_merged_data, test_merged_data)


if __name__ == "__main__":
    t4()
