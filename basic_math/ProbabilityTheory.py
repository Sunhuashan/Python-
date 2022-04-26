# 预备数学知识--概率论
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones(1, 6) / 6  # 概率
counts = multinomial.Multinomial(10000, fair_probs).sample()  # 随机取样
print(counts / 10000)  # 频率 Tensor(1*6)

counts = multinomial.Multinomial(10, fair_probs).sample((1000,))  # Tensor(1000*1*6)
cumsum_counts = counts.cumsum(axis=0)  # 元素累积和 Tensor(1000*1*6)
estimate = cumsum_counts / cumsum_counts.sum(axis=2, keepdim=True)  # 频率 Tensor(1000*1*6) / Tensor(1000*1*1)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimate[:, :, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
plt.show()
