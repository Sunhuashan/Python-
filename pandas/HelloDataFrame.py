# Pytorch入门--pandas-DataFrame
#
# REMEMBER
# Import the package, aka import pandas as pd
#
# A table of data is stored as a pandas DataFrame
#
# Each column in a DataFrame is a Series
#
# You can do things by applying a method to a DataFrame or Series
import pandas as pd
df = pd.DataFrame(  # 初识 DataFrame,使用 字典对象 手动创建
    {
        "姓名": [
            "张三",
            "李四",
            "王二"
        ],
        "年龄": [18, 20, 35],
        "性别": ["男", "女", "男"],
    }
)
print(df)

# 查看某列，返回的为 Series:
# A pandas Series has no column labels,
# as it is just a single column of a DataFrame.
# A Series does have row labels.
print(df["年龄"])

age = pd.Series([19, 20, 21], name="年龄")  # 一个 pandas Series
print(age)

print(df["年龄"].max())  # 对 Series 的操作，即对每列数据的操作

print(df.describe())  # DataFrame 数值列 相关统计信息的描述，如最值，均值，标准差等


