'''
处理问题4预测结果只保留一位
'''
import pandas as pd

if __name__ == '__main__':
    result = pd.read_excel('test_data3.xlsx')
    # 对预测值这一列只保留小数点后1位
    result['预测值'] = result['预测值'].apply(lambda x: round(x, 1))
    result.to_excel('test_data4.xlsx', index=False)
