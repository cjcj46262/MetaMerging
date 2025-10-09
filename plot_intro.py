import matplotlib.pyplot as plt
import numpy as np

# 方法名称
methods = ["Weight Averaging", "Task Arithmetic", "Ties-Merging", "AdaMerging", "MetaMerging"]

# 有/无 adapter 的平均分
avg_without_adapters = [65.8, 70.1, 73.6, 80.1, 67.8]
avg_with_adapters    = [80.0, 80.9, 83.1, 86.1, 87.2]

x = np.arange(len(methods))
width = 0.35

plt.figure(figsize=(8, 5))

# 柱状图
bars1 = plt.bar(x - width/2, avg_without_adapters, width, 
                label="unified model", color="#D79B00", edgecolor="black", alpha=0.87)
bars2 = plt.bar(x + width/2, avg_with_adapters, width, 
                label="w/ adapters", color="#B85450", edgecolor="black", alpha=0.87)

# 添加数值标签
for bar in bars1 + bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, 
             f"{height:.1f}", ha="center", va="bottom", fontsize=14)

# 美化图表
plt.xticks(x, methods, rotation=20, fontsize=18)
plt.ylabel("Average Accuracy (%)", fontsize=18)
plt.ylim(40, 100)   # y 轴从 50 到 100
plt.yticks(fontsize=11)
plt.legend(fontsize=15, frameon=True, shadow=True, edgecolor="gray", loc='upper left')
plt.grid(axis="y", linestyle="--", alpha=0.6)

# 边框样式
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.tight_layout()

# 保存为 PDF
plt.savefig("./figures/intro1.pdf", bbox_inches='tight', pad_inches=0)
