# PyTorch入门-构建数据集

import torch
import pandas as pd
import os

os.makedirs(os.path.join('..', 'my_data'), exist_ok=True)  # 创建目录
data_file = os.path.join('..', 'my_data', 'house_tiny.csv')  # 创建CSV文件使用逗号分隔数据)
with open(data_file, 'w') as f:  # 写入文件内容
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,123456\n')
    f.write('NA,NA,123312\n')

df = pd.read_csv(data_file)  # pandas读入CSV文件 => DataFrame
print(df)

inputs, outputs = df.iloc[:, 0:-1], df.iloc[:, -1]  # 划分输入，输出
inputs = inputs.fillna(inputs.mean())  # 输入集的缺省值处理--设为均值
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)  # 缺省值处理--将 NAN 视为一类
print(inputs)

# DataFrame 转换为张量，便于计算
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)  # DataFrame.values 返回 np.ndarray
print(f"Inputs convert to tensor: \n {x}")
print(f"Outputs convert to tensor: \n {y}")

# 练习1,删除缺失值最多的一列
bool_df = pd.isna(df)  # 返回 bool DataFrame，即元素: NAN=>True;其他=>False
print(bool_df.sum())  # 返回 Series
dic = bool_df.sum().to_dict()  # 转为字典
print(dic)
max_key = max(dic, key=dic.get)  # 获取字典值最大的键
print(max_key)
new_df = df.drop(columns=max_key)  # 删除对应列
print(new_df)

# 练习2,将预处理后的数据转为tensor
# DataFrame.values => np.ndarray
# torch.tensor(np.ndarray) => tensor