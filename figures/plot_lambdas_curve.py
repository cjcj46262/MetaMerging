import numpy as np
import re
import matplotlib.pyplot as plt

# 读取日志文件
logfile = "/home/cj/EMR_Merging/merge_vit/logs/ViT-B-32/log_2025_09_20_23_43_08_1_0.005_0.1_0.001.txt"

steps = []
lambdas = []

pattern = re.compile(r"metastep (\d+) .* lambdas: tensor\(\[\[1.0000, (.*?)\]\]")

with open(logfile, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            step = int(match.group(1))
            values = [float(x) for x in match.group(2).split(",")]
            steps.append(step)
            lambdas.append(values)

# 转置，得到每个 lambda 的序列
lambdas = list(zip(*lambdas))

def ema(data, alpha=0.1):
    """指数加权移动平均 (EMA)"""
    smoothed = []
    for x in data:
        if not smoothed:
            smoothed.append(x)
        else:
            smoothed.append(alpha * x + (1 - alpha) * smoothed[-1])
    return smoothed

# 绘图
plt.figure(figsize=(7,4))
exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
alpha = 0.02  # 平滑系数，越小越平滑，越大越接近原始曲线
for i, lam in enumerate(lambdas):
    smoothed = ema(lam, alpha=alpha)
    plt.plot(steps, smoothed, label=exam_datasets[i])

# 加一条横线 y=0.125
plt.axhline(y=0.125, color="black", linestyle="--", linewidth=1)

# 在横线右边标注星号和文字
plt.text(steps[0], 0.125 - 0.002, "0.125 (Weight Averaging)",
         color="black", fontsize=15,
         ha="left", va="top")  # 改成 top

plt.xlabel("Meta-Unpdating Step",fontsize=17)
plt.ylabel("Lambda Value",fontsize=17)
# plt.title("EMA Smoothed Lambdas during Training")
plt.legend(
    loc="upper center",          # 图例放在上方
    bbox_to_anchor=(0.5, 1.28),  # 相对位置，(0.5, 1.15) 表示居中并往上移
    ncol=4,                      # 一行 4 列
    fontsize=12.5,                 # 字体大小可以调
    # frameon=True,                # 去掉边框（可选）
    # shadow=True,
    # edgecolor="gray"
)
plt.grid(False)
plt.tight_layout()
plt.savefig("./figures/lambdas_curve_ema.pdf", bbox_inches='tight', pad_inches=0)
