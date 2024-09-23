'''
问题一:先取磁通密度的时域特征，峰值、峰峰值等（时域图），再取磁通密度经过傅里叶变换转频域特征，提取主频、能量频域特征，一起合并
对合并结果进行标准化，再PCA降维(看图选降到多少维)，评价降维后的。
降维后数据送到决策树模型进行训练和验证，然后到目标文件进行预测波形三分类
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mkl_fft import fft
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from logger import logger
from sklearn.metrics import ConfusionMatrixDisplay


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
    temperature = sheet.iloc[:, 0]  # 温度
    frequency = sheet.iloc[:, 1]  # 频率
    core_loss = sheet.iloc[:, 2]  # 磁芯损耗
    waveform_type = sheet.iloc[:, 3]  # 励磁波形类型
    flux_density = sheet.iloc[:, 4:]  # 磁通密度
    # 通过正态概率图判断sheet的磁通密度的符合正态分布
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.figure(figsize=(12, 8))
    # plt.hist(flux_density, bins=50)
    # plt.title('磁通密度正态分布')
    # plt.xlabel('磁通密度')
    # plt.ylabel('出现频率')
    # plt.show()
    return temperature, frequency, core_loss, waveform_type, flux_density


# 不同波形类型下的磁通密度变化
def get_magnetic_density_feature(
        temperature, frequency, core_loss, waveform_type, flux_density, data):
    unique_waveforms = waveform_type.unique()
    features = {}
    data = data.iloc[:, 4:]


# 计算时域特征:data磁通密度
def extract_time_domain_features(data):
    features = {}
    # 计算时域特征
    features['mean'] = np.mean(data, axis=1)
    features['std'] = np.std(data, axis=1)
    features['peak'] = np.max(data, axis=1)
    features['peak_to_peak'] = np.ptp(data, axis=1)
    features['skewness'] = skew(data, axis=1)
    features['kurtosis'] = kurtosis(data, axis=1)
    features['waveform_factor'] = np.abs(
        np.diff(np.unwrap(np.angle(fft(data))), axis=1)).mean(axis=1)  # 确保输出是一维 波形因子
    features['rms'] = np.sqrt(np.mean(data ** 2, axis=1))  # 均方根
    return pd.DataFrame(features)


# 计算频域特征
def extract_frequency_domain_features(data):
    features = {}
    # 对每行数据进行FFT
    fft_data = fft(data)
    # 提取频域特征（主频、能量等）
    # 分别取fft第一个
    features['fft_0'] = np.real(fft_data[:, 0])
    features['fft_1'] = np.real(fft_data[:, 1])
    features['fft_2'] = np.real(fft_data[:, 2])
    features['dominant_frequency'] = np.argmax(np.abs(fft_data), axis=1)
    features['spectral_energy'] = np.sum(np.abs(fft_data) ** 2, axis=1)
    return pd.DataFrame(features)


# 时域和频域特征合并
def merge_time_and_frequency_features(data):
    freq_features = extract_time_domain_features(data.iloc[:, 4:].values)
    time_features = extract_frequency_domain_features(data.iloc[:, 4:].values)
    # print(freq_features.shape)
    # plot_time_domain_features(freq_features)
    return pd.concat([time_features, freq_features], axis=1)


def plot_time_domain_features(features):
    data = pd.read_excel('excel/merge_train1.xlsx')
    # 删除材料列
    data = data.drop('材料', axis=1)
    # 假设`analysis_magnetic_density`函数返回正确的数据
    _, _, _, waveform_type, _ = analysis_magnetic_density(data)
    print(waveform_type)
    unique_waveforms = waveform_type.unique()

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制时域特征
    plt.figure(figsize=(12, 8))
    num_plots = len(unique_waveforms)
    num_cols = 2
    num_rows = int(np.ceil(num_plots / num_cols))

    for i, waveform in enumerate(unique_waveforms):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.title(f'波形: {waveform}')
        # 根据波形类型筛选特征
        waveform_features = features[waveform_type == waveform]
        mean_values = waveform_features['mean']
        std_values = waveform_features['std']
        peak_values = waveform_features['peak']
        peak_to_peak_values = waveform_features['peak_to_peak']
        skewness_values = waveform_features['skewness']
        kurtosis_values = waveform_features['kurtosis']
        rms_values = waveform_features['rms']
        waveform_factor_values = waveform_features['waveform_factor']
        plt.plot(mean_values, label='均值')
        plt.fill_between(range(len(mean_values)),
                         [m - s for m, s in zip(mean_values, std_values)],
                         [m + s for m, s in zip(mean_values, std_values)],
                         alpha=0.2)
        plt.plot(peak_values, label='峰值', linestyle='--')
        plt.plot(peak_to_peak_values, label='峰峰值', linestyle=':')
        plt.plot(skewness_values, label='偏度', linestyle='-.')
        plt.plot(kurtosis_values, label='峰度', linestyle='-.')
        plt.plot(rms_values, label='均方根', linestyle='-.')
        plt.plot(waveform_factor_values, label='波形因子', linestyle='-.')
        plt.xlabel('样本点')
        plt.ylabel('磁通密度 (T)')
        plt.legend()

    plt.tight_layout()
    plt.show()


# PCA降维
def pca_reduction(data, test_data):
    features = merge_time_and_frequency_features(data)
    print(features.shape)
    test_features = merge_time_and_frequency_features(test_data)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    test_features = scaler.transform(test_features)
    # print('数据原始维度是%d维' % (features.shape[1]))
    # 观察累计解释方差比随主成分数量增加的变化曲线，选择“肘部”位置作为主成分数量。为2
    pca = PCA().fit(features)
    # 加上标记点
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.title('累计解释方差比随主成分数量增加的变化曲线')
    # plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    # plt.xlabel('主成分数量')
    # plt.ylabel('累计解释方差比')
    # plt.show()
    # PCA降维
    pca = PCA(n_components=7)  # 自动按照内部函数的选择维度方法
    train_pca_feature = pca.fit_transform(features)
    test_pca_feature = pca.transform(test_features)
    print(pca.explained_variance_ratio_.sum())
    # 所有保留主成分的方差占比之和为0.999997647807451
    return train_pca_feature, test_pca_feature


# 三分类
def three_classification(data, pca_features, test_features):
    # sheet1对应测试集
    test_features = pd.DataFrame(test_features)
    # test_features = test_features.iloc[60:80]
    print(test_features)
    newX = pca_features
    data['励磁波形'] = data['励磁波形'].map({'正弦波': 1, '三角波': 2, '梯形波': 3})
    # 分离X和Y
    X = data[data.columns[4:]]
    Y = data['励磁波形']
    X_train, X_test, Y_train, Y_test = train_test_split(
        newX, Y, test_size=0.3, random_state=42)
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
        # 交叉验证
        cv_scores = cross_val_score(model, X_train, Y_train, cv=5)
        cv_mse = -cv_scores.mean()
        print(f'{name}模型在交叉验证中的均方误差：{cv_mse}')
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
        # 绘制混淆矩阵，颜色为科研配色
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
        plt.rcParams['axes.unicode_minus'] = False
        cm = confusion_matrix(Y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=model.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'{name} 混淆矩阵')
        plt.show()
        # 把颜色调为科研配色
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        # disp.plot()
        # plt.title(f'{name} 混淆矩阵')
        # plt.show()

    for name, model in models.items():
        print(test_features.shape)
        # 使用训练好的模型进行预测
        test_predictions = model.predict(test_features)
        # 模型评估
        print(f'这个模型:{name}的预测结果为{test_predictions}')
        # 输出分类结果到附件四（保存为CSV文件）
        # output = pd.DataFrame({'样本序号': test_data.iloc[:, 0], '波形分类': test_predictions})
        # output.to_csv('附件四.csv', index=False)


def t1():
    data = pd.read_excel('excel/merge_train1.xlsx')
    # 删除材料列
    data = data.drop('材料', axis=1)
    test_data = read_excel('excel/test2.xlsx', '测试集')
    temperature, frequency, core_loss, waveform_type, flux_density = analysis_magnetic_density(
        data)
    # features = get_magnetic_density_feature(temperature, frequency, core_loss, waveform_type, flux_density, sheet1)
    train_pca_features, test_pca_features = pca_reduction(data, test_data)
    three_classification(data, train_pca_features, test_pca_features)


if __name__ == '__main__':
    logger.info('开始执行')
    t1()
    logger.info('执行结束')
