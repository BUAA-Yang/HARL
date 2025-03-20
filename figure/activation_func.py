import numpy as np
import matplotlib.pyplot as plt
# 设置 Matplotlib 显示中文字
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def selu(x, alpha=1.67326, scale=1.0507):
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

# 定义输入范围
x = np.linspace(-10, 10, 500)

# 计算激活函数值
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_selu = selu(x)

# 创建子图
fig, axes = plt.subplots(1, 5, figsize=(10, 2.2))

# 绘制每个激活函数的子图
axes[0].plot(x, y_sigmoid, label="Sigmoid", color="blue")
axes[0].set_title("Sigmoid")

axes[1].plot(x, y_tanh, label="Tanh", color="orange")
axes[1].set_title("Tanh")

axes[2].plot(x, y_relu, label="ReLU", color="green")
axes[2].set_title("ReLU")

axes[3].plot(x, y_leaky_relu, label="Leaky ReLU", color="red")
axes[3].set_title("Leaky ReLU")

axes[4].plot(x, y_selu, label="SELU", color="purple")
axes[4].set_title("SELU")

# 设置每个子图的坐标轴和标签
for i, ax in enumerate(axes):
    ax.axhline(0, color='black', linewidth=1)  # 实线
    ax.axvline(0, color='black', linewidth=1)  # 实线
    if i == 0:  # 仅在最左侧显示 y 轴标题
        ax.set_ylabel("输出")
    if i == 2:
        ax.set_xlabel("输入")

# 调整布局并显示图像
plt.tight_layout()
plt.savefig("activation_functions.pdf")
