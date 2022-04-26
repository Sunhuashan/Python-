import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()  # 一个有坐标轴的图
ax.plot([1, 2, 3, 4], [2, 4, 0, 5])
plt.show()


fig = plt.figure()  # 一个空图
fig, ax = plt.subplots()  # 一个带有坐标轴的图
# fig, ax = plt.subplots(2, 2)  # 一个带有2*2网格坐标轴的图

b_matrix = np.matrix([1, 3], [3, 4])  # 绘图函数期望的数据类型为numpy.array
b_asarray = np.asarray(b_matrix)