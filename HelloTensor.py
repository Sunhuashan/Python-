# Pytorch入门程序-Tensor(张量)
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
print(f"Matrix multipliction of tensor: \n {tensor}")
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
z = torch.rand_like(x)   # dim,即dimension 维度或理解为轴
torch.cat((x, y), dim=0, out=z)  # 3*4 => 6*4
print(z)
torch.cat((x, y), dim=1, out=z)  # 3*4 => 3*8
print(z)

x = torch.rand(3, 4, 8, 7)  # 拼接条件：除了选择的 dim 可以不一致，其余必须一致
y = torch.ones(3, 4, 1, 7)  # 如，3*4*8*7 的张量与 3*4*1*7 的张量只能在 dim=2 轴拼接
z = torch.cat([x, y], dim=2)  # 如，3*4 的张量与 4*3 的张量永远无法拼接
print(f"Shape:  {z} \n {z.shape}")

x = torch.tensor([[1, 2, 3], [3, 4, 5]])  # 元素值的和
print(x.sum())