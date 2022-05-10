# PyTorch入门程序-Tensor(张量)
import torch
import numpy

# PyTorch官方文档示例
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)  # 直接从 list对象 中创建张量

np_array = numpy.array(data)  # 从Numpy数组转化成张量
x_np = torch.from_numpy(np_array)

x_zeros = torch.zeros_like(x_data)  # 从张量转化成张量，重写数据为0,保留形状
print(f'Zeros Tensor: \n {x_zeros} \n')

x_ones = torch.ones_like(x_data)  # 从张量转化成张量，重写数据为1,保留形状
print(f'Ones Tensor: \n {x_ones} \n')

x_rand = torch.rand_like(x_data, dtype=torch.float)  # 从张量转化成张量，随机[0-1]重写数据,保留形状
print(f'Random Tensor: \n {x_rand} \n')

shape = (3, 4,)  # 用元组来定义张量的维度
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

print(f'Tensor data: {rand_tensor.data}')  # data属性
print(f'Shape of tensor: {rand_tensor.shape}')  # shape属性
print(f'Datatype of tensor: {rand_tensor.dtype}')  # dtype属性
print(f'Device tensor is stored on: {rand_tensor.device}\n')  # device属性

tensor = torch.rand(4, 4)  # 张量的相关操作
print(tensor)
print(f'First row: {tensor[0]}')  # 获取行
print(f'Last row: {tensor[-1]}')
print(f'First column: {tensor[:, 0]}')  # 获取列
print(f'Last column: {tensor[:, -1]}\n')
tensor[:, 1] = 0  # 设置列的值
print(f'Column two set zeros: \n {tensor}\n')
tensor1 = torch.cat([tensor, tensor], dim=0)  # 张量连接
tensor2 = torch.cat([tensor, tensor], dim=1)
print(f'Cat in column: \n {tensor1} \n')
print(f'Cat in row: \n {tensor2} \n')

tensor = torch.rand(2, 2)  # 张量的matrix multiplication(矩阵乘法)
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.zeros_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
print(f"Matrix multiplication of tensor: \n {tensor}")
print(f"Result one: \n {y1}")
print(f"Result two: \n {y2}")
print(f"Result three: \n {y3} \n")

tensor = torch.rand(2, 2)  # 张量的element-wise product(元素级乘法)
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(f"Element-wise product of tensor: \n {tensor}")
print(f"Result one: \n {z1}")
print(f"Result two: \n {z2}")
print(f"Reesult three: \n {z3}")

tensor = torch.rand(1, 1)  # 0维张量与python数字变量的转换
num = tensor.item()
print(f"Tensor convert to numerical value: \n {tensor} \n {num}{type(num)}")

tensor = torch.rand(4, 4)  # 张量的in-place operation(就地操作)，以 "_" 结尾,覆盖历史数据，不建议使用
print(f"In-palce add of tensor: \n {tensor}")
tensor.add_(5)
print(f"Result: \n {tensor}")

tensor = torch.ones(5)  # 张量 与 Numpy数组 共享地址空间
array = tensor.numpy()
print(f"Tensor: {tensor} \n Numpy array: {array}")
tensor.add_(2)
print(f"Tensor: {tensor} \n Numpy array: {array}")

array = numpy.ones(6)
tensor = torch.from_numpy(array)
print(f"Tensor: {tensor} \n Numpy array: {array}")
numpy.add(array, 1, out=array)
print(f"Tensor: {tensor} \n Numpy array: {array}")

# d2l 文档中示例
x = torch.arange(12)  # 使用 arange 创建行向量
print(x)
print(x.shape)  # 张量形状
print(x.size())  # 张量大小
print(x.numel())  # 张量元素个数
print(x.reshape(3, 4))  # 使用 reshape 改变形状
print(x.reshape(-1, 4))  # 使用 -1 自动计算
print(x.reshape(3, -1))

x = torch.ones(3, 4, 5)  # 同上
x = torch.zeros(3, 4, 5)
x = torch.randn(3, 4, 5)  # 从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样
x = torch.tensor([[[1, 2, 3, 4],
                   [1, 3, 4, 5],
                   [2, 4, 4, 7]],

                  [[1, 2, 3, 4],
                   [1, 3, 4, 5],
                   [2, 4, 4, 7]]])  # 2*3*4的 list 对象转换成 tensor
print(x.shape)

x = torch.tensor([1, 2, 3, 6])  # element wise 同形状张量的元素级运算
y = torch.tensor([2, 4, 6, 8])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)  # 幂指运算
print(torch.exp(x))  # e^n 计算
print(x == y)  # 练习1
print(x > y)
print(x < y)

x = torch.rand(3, 4)  # 张量的连接 con'cat'enate
y = torch.ones(3, 4)
z = torch.rand_like(x)   # dim,即dimension理解为轴
torch.cat((x, y), dim=0, out=z)  # 3*4 => 6*4
print(z)
torch.cat((x, y), dim=1, out=z)  # 3*4 => 3*8
print(z)

x = torch.rand(3, 4, 8, 7)  # 拼接条件：除了选择的 dim 可以不一致，其余必须一致
y = torch.ones(3, 4, 1, 7)  # 如，3*4*8*7 的张量与 3*4*1*7 的张量只能在 dim=2 轴拼接
z = torch.cat([x, y], dim=2)  # 如，3*4 的张量与 4*3 的张量永远无法拼接
print(f"Shape:  {z} \n {z.shape}")

x = torch.tensor([[1, 2, 3], [3, 4, 5]])  # 元素值的和 n*m => 1*1
print(x.sum())

x = torch.tensor([[1],
                  [2],
                  [3]])  # 3*1 => 3*2
y = torch.tensor([4, 5])  # 1*2 => 3*2
print(x + y)  # 不同形张量自动使用广播机制(复制已有轴的元素)转为同形张量

x_broc = torch.tensor([[1, 1],
                       [2, 2],
                       [3, 3]])
y_broc = torch.tensor([[4, 5],
                       [4, 5],
                       [4, 5]])
print(x_broc + y_broc)  # 使用广播机制后的元素级加法
#  广播机制中不匹配的维数中必须是 1 => n,从1维 升到 n维
#  例如 2*3 => 3*3 是不能实现的，系统不知道该复制哪一个（2 => 3）
#======================================================================
# 关于广播机制的一些补充：
# 例如两个tensor的shape分别为(8, 1, 6, 5)和 (7, 1, 5)，那么是否可以广播呢？
# 做右对齐, 空缺的位置假想为1:
# 8, 1, 6, 5
# 1, 7, 1, 5
# 按照以上规则得出是可以广播的，操作结果的shape应为(8, 7, 6, 5)
#======================================================================
x = torch.ones(3, 4, 1)  # 3*4*1 => 3*4*5
y = torch.ones(1, 1, 5)  # 1*1*5 => 3*4*5
z = x + y
print(z.shape)  # 3*4*5

x = torch.rand(3, 4, 5)  # 索引与切片
print(x[0])  # 0-轴第一个元素
print(x[-1])  # 0-轴最后一个元素
print(x[0:2])  # 0-轴第一到第二个元素,即左闭右开！
print(x[1, 0, 2])  # 0-轴第二个，1-轴第一个，2-轴第三个元素
x[1:3, -1, :] = 0  # 0-轴第二到第三个，1-轴最后一个，2-轴全部元素 赋值
print(x)

Y = torch.ones(2, 4)
X = torch.ones(2, 4)
before = id(Y)
Y = X + Y  # 自动开辟新内存，将会造成空间浪费
after = id(Y)
print(f"before: {before} \n after: {after} \n")

before = id(Y)
Y[:] = X + Y  # 切片表示法避免重新分配内存空间
after = id(Y)
print(f"before: {before} \n after: {after} \n")

X = torch.ones(3, 4)  # numpy数组与tensor之间的转换
np_array = X.numpy()
tensor = torch.from_numpy(np_array)
print(type(np_array))
print(type(tensor))

Y = torch.rand(1, 1)  # 0维tensor 与 python对象 之间的转换
print(Y)
print(Y.item())  # 调用自身函数
print(float(Y))  # 强转
print(int(Y))  # 强转

# 张量的维度是指张量具有的轴数(标量是0维张量，向量是1维张量，矩阵是2维张量...)，
# 而张量的某个轴的维度是指该轴的长度，即该轴的元素个数。
# To clarify, we use the dimensionality of a vector or an axis to refer to its length,
# i.e., the number of elements of a vector or an axis.
# However, we use the dimensionality of a tensor to refer to the number of axes that a tensor has.
# In this sense, the dimensionality of some axis of a tensor will be the length of that axis.

tensor = torch.ones(3, 4, 5)  # 张量降维求和
sum_axis0 = tensor.sum(axis=0)  # 3*4*5 => 1*4*5/4*5
sum_axis1 = tensor.sum(axis=1)  # 3*4*5 => 3*1*5/3*5
sum_axis2 = tensor.sum(axis=2)  # 3*4*5 => 3*4*1/3*4
sum_axis0_1 = tensor.sum(axis=[0, 1])  # 3*4*5 => 1*1*5/5(向量）
sum_axis0_1_2 = tensor.sum()  # 3*4*5 => 1*1*1/1(标量)
print(tensor)
print(sum_axis0)
print(sum_axis1)
print(sum_axis2)
print(sum_axis0_1)
print(sum_axis0_1_2)

tensor = torch.rand(3, 4, 5)  # 张量降维求均值
mean_axis0_1 = tensor.mean(axis=[0, 1])  # 3*4*5 => 5
print(tensor)
print(mean_axis0_1)
print(tensor.sum(axis=[0, 1]) / (tensor.shape[0]*tensor.shape[1]))  # 降维求和后再算均值

tensor = torch.ones(3, 4, 5)  # 张量不降维求和(保持轴数不变，对比：[[1], [2]] VS [1, 2] ,前者保留 1-轴)
sum_keepdims = tensor.sum(axis=-1, keepdims=True)  # 待定：使用 keepdims 关键字只能保证沿着最后一个轴时不降维？
print(tensor.shape)
print(sum_keepdims.shape)

print(tensor.cumsum(axis=2))  # 沿某轴，计算累积和

x = torch.tensor([1, 3, 5, 2])
y = torch.arange(4)
print(x)
print(y)
print(torch.dot(x, y))  # 向量间的内积/点积 元素对应相乘的和
print((x*y).sum())  # 内积的另一种实现

A = torch.rand(4, 5, dtype=float)  # Matrix-Vector Products 矩阵向量积
v = torch.arange(5, dtype=float)
print(A)
print(v)
print(torch.mv(A, v))  # mv--matrix vector

A = torch.rand(4, 5, dtype=float)  # Matrix-Matrix Products
B = torch.rand(5, 8, dtype=float)
print(torch.mm(A, B))  # => 4*8

# 向量的 norm 范数/模
v = torch.arange(2, dtype=float)
print(v.abs().sum())  # 向量的 L1范数
q = torch.tensor([3, 4], dtype=float)
print(q.norm())  # 向量的 L2范数

# 练习4，5
x = torch.rand(7, 3, 4)  # len(tensor) = 0-轴的维度
print(len(x))

# 练习6
# 由于沿某轴降维求和后，轴数减1，无法相除
# 可以使 keepdims=True 保持维数,配合广播机制成功运行
A = torch.rand(3, 4)
print(A / A.sum(axis=1, keepdims=True))

# 练习7
x = torch.ones(2, 3, 4)
print(x.sum(axis=0).shape)  # 3*4
print(x.sum(axis=1).shape)  # 2*4
print(x.sum(axis=2).shape)  # 2*3


