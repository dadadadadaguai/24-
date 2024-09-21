import numpy as np
import pandas as pd
from mkl_fft import fft
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from logger import logger


# 读取全部sheet
def read_all_sheet(xlsx_path: str):
    xls = pd.read_excel(xlsx_path)
    sheet_to_df_map = {sheet_name: xls.parse(
        sheet_name) for sheet_name in xls.sheet_names}
    return sheet_to_df_map


# 读取xlsx的sheet
def read_excel(xlsx_path: str, sheet_name: str):
    xls = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    return xls


# 读取磁通密度
def read_magnetic_density(xlsx_path: str, sheet_name: str):
    all_sheet = read_all_sheet(xlsx_path)
    return all_sheet[sheet_name].iloc[:, 4:1029]


# 分析磁通密度的分布特征
def analysis_magnetic_density(sheet):
    # 通过正态概率图判断sheet的磁通密度的符合正态分布
    # plt.figure(figsize=(12, 6))
    # plt.hist(sheet.iloc[:, 4:], bins=50, density=True)
    # plt.title('Histogram of Flux Density')
    # plt.show()
    # Shapiro-Wilk 检验
    # flux_density = sheet.iloc[:, 4:]
    # # 对每一列应用 Kolmogorov-Smirnov 检验
    # results_kstest = {}
    # for column in flux_density:
    #     data = flux_density[column]
    #
    #     # 使用正态分布的累积分布函数作为基准
    #     stat, p = kstest(data, 'norm')
    #     results_kstest[column] = (stat, p)
    #
    # # 输出结果
    # for col, result in results_kstest.items():
    #     print(f'Kolmogorov-Smirnov 统计值: {result[0]}, P值: {result[1]}')
    temperature = sheet.iloc[:, 0]  # 温度
    frequency = sheet.iloc[:, 1]  # 频率
    core_loss = sheet.iloc[:, 2]  # 磁芯损耗
    waveform_type = sheet.iloc[:, 3]  # 励磁波形类型
    flux_density = sheet.iloc[:, 4:]  # 磁通密度
    return temperature, frequency, core_loss, waveform_type, flux_density


# 不同波形类型下的磁通密度变化
def get_magnetic_density_feature(temperature, frequency, core_loss, waveform_type, flux_density, data):
    unique_waveforms = waveform_type.unique()
    features = {}
    data = data.iloc[:, 4:]


# 计算时域特征
def extract_time_domain_features(data):
    features = {}
    # 计算时域特征
    features['mean'] = np.mean(data, axis=1)
    features['peak'] = np.max(data, axis=1)
    features['peak_to_peak'] = np.ptp(data, axis=1)
    features['skewness'] = skew(data, axis=1)
    features['kurtosis'] = kurtosis(data, axis=1)
    return pd.DataFrame(features)


# 计算频域特征
def extract_frequency_domain_features(data):
    features = {}
    # 对每行数据进行FFT
    fft_data = fft(data)
    # 提取频域特征（主频、能量等）
    features['dominant_frequency'] = np.argmax(np.abs(fft_data), axis=1)
    features['spectral_energy'] = np.sum(np.abs(fft_data) ** 2, axis=1)
    return pd.DataFrame(features)


# 时域和频域特征合并
def merge_time_and_frequency_features(data):
    freq_features = extract_time_domain_features(data.iloc[:, 4:].values)
    time_features = extract_frequency_domain_features(data.iloc[:, 4:].values)
    return pd.concat([time_features, freq_features], axis=1)
    # 可视化分析
    ## 1. 时域特征可视化
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.figure(figsize=(12, 8))
    # num_plots = len(unique_waveforms)
    # num_cols = 2
    # num_rows = int(np.ceil(num_plots / num_cols))
    #
    # for i, waveform in enumerate(unique_waveforms):
    #     plt.subplot(num_rows, num_cols, i + 1)
    #     plt.title(f'Waveform: {waveform}')
    #
    #     mean_values = features[waveform]['mean']
    #     std_values = features[waveform]['std']
    #     peak_values = features[waveform]['peak']
    #     peak_to_peak_values = features[waveform]['peak_to_peak']
    #     skewness_values = features[waveform]['skewness']
    #     kurtosis_values = features[waveform]['kurtosis']
    #
    #     plt.plot(mean_values, label='均值')
    #     plt.fill_between(range(len(mean_values)),
    #                      [m - s for m, s in zip(mean_values, std_values)],
    #                      [m + s for m, s in zip(mean_values, std_values)],
    #                      alpha=0.2)
    #
    #     plt.plot(peak_values, label='峰值', linestyle='--')
    #     plt.plot(peak_to_peak_values, label='峰峰值', linestyle=':')
    #     plt.plot(skewness_values, label='偏度', linestyle='-.')
    #     plt.plot(kurtosis_values, label='峰度', linestyle='-.')
    #
    #     plt.xlabel('Sample Points')
    #     plt.ylabel('磁通密度 (T)')
    #     plt.legend()
    #
    # plt.tight_layout()
    # plt.show()


# PCA降维
def pca_reduction(data, test_data):
    features = merge_time_and_frequency_features(data)
    test_features = merge_time_and_frequency_features(test_data)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    test_features = scaler.transform(test_features)
    # print('数据原始维度是%d维' % (features.shape[1]))
    # 观察累计解释方差比随主成分数量增加的变化曲线，选择“肘部”位置作为主成分数量。为2
    # pca = PCA().fit(features)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('Number of Components')
    # plt.ylabel('Cumulative Explained Variance')
    # plt.show()
    # PCA降维
    pca = PCA(n_components=5)  # 自动按照内部函数的选择维度方法
    train_pca_feature = pca.fit_transform(features)
    test_pca_feature = pca.transform(test_features)
    # print(pca.explained_variance_ratio_.sum())  所有保留主成分的方差占比之和为0.999997647807451
    return train_pca_feature, test_pca_feature


# 三分类
def three_classification(data, pca_features, test_features):
    # sheet1对应测试集
    test_features = pd.DataFrame(test_features)
    test_features = test_features.iloc[60:80]
    print(test_features)
    newX = pca_features
    data['励磁波形'] = data['励磁波形'].map({'正弦波': 1, '三角波': 2, '梯形波': 3})
    # 分离X和Y
    X = data[data.columns[4:]]
    Y = data['励磁波形']
    X_train, X_test, Y_train, Y_test = train_test_split(newX, Y, test_size=0.3, random_state=42)
    # 定义模型
    models = {
        '逻辑回归': LogisticRegression(),
        '决策树': DecisionTreeClassifier(random_state=42),
        '随机森林': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100),
        'GBDT': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    }

    # 全局ROC曲线数据
    roc_data = []

    # 处理每个模型
    for name, model in models.items():
        # 拟合模型
        model.fit(X_train, Y_train)
        # 预测
        y_pred = model.predict(X_test)
        # 模型评估
        acc = accuracy_score(Y_test, y_pred, )
        rec = recall_score(Y_test, y_pred, average='micro')
        f1 = f1_score(Y_test, y_pred, average='micro')
        classification = classification_report(Y_test, y_pred)
        print(f'{name}模型在训练集上的预测结果：')
        print(f'{name}模型评价结果：')
        print("ACC", acc)
        print("REC", rec)
        print("F-score", f1)
        print(classification)

    for name, model in models.items():
        print(test_features.shape)
        # 使用训练好的模型进行预测
        test_predictions = model.predict(test_features)
        # 模型评估
        print(test_predictions)
        # 输出分类结果到附件四（保存为CSV文件）
        # output = pd.DataFrame({'样本序号': test_data.iloc[:, 0], '波形分类': test_predictions})
        # output.to_csv('附件四.csv', index=False)


def t1():
    data = read_excel('excel/train1.xlsx', '材料4')
    test_data = read_excel('excel/test2.xlsx', '测试集')
    print(test_data.shape)
    temperature, frequency, core_loss, waveform_type, flux_density = analysis_magnetic_density(data)
    # features = get_magnetic_density_feature(temperature, frequency, core_loss, waveform_type, flux_density, sheet1)
    train_pca_features, test_pca_features = pca_reduction(data, test_data)
    three_classification(data, train_pca_features, test_pca_features)


if __name__ == '__main__':
    logger.info('开始执行')
    t1()
    logger.info('执行结束')
