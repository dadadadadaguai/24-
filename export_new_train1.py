import pandas as pd


def export_xlsx():
    #现在train1.xlsx分为4个sheet,代表4种材料的数据，现在希望合并同时添加一列标记他们的材料类别
    sheet_names = {
        '材料1': '1',
        '材料2': '2',
        '材料3': '3',
        '材料4': '4'
    }
    # 创建一个空的DataFrame来存储合并后的数据
    merged_df = pd.DataFrame()

    # 遍历每个sheet
    for sheet_name, material_type in sheet_names.items():
        # 读取当前sheet的数据
        df = pd.read_excel('excel/train1.xlsx', sheet_name=sheet_name)

        # 添加一个新的列，表示材料类别
        df['材料'] = material_type

        # 将当前sheet的数据追加到合并的DataFrame中
        merged_df = pd.concat([merged_df, df])

    # 将合并后的数据保存到新的Excel文件中
    merged_df.to_excel('excel/merged_data.xlsx', index=False)

if __name__ == '__main__':
    export_xlsx()
