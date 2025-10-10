import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# 数据集名称
datasets = ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"]

# 五种方法
methods = ["Weight Average", "Task Arithmetic", "Ties-Merging", "AdaMerging", "Ours"]

# w/o adapters 数据
w_o_adapter = np.array([
    [65.3, 63.8, 64.8, 64.5, 56.8],
    [63.4, 62.1, 62.9, 68.1, 60.3],
    [71.4, 72.0, 74.3, 79.2, 57.3],
    [71.7, 77.6, 78.9, 93.8, 94.5],
    [64.2, 74.4, 83.1, 87.0, 87.2],
    [52.8, 65.1, 71.4, 91.9, 40.2],
    [87.5, 94.0, 97.6, 97.5, 96.3],
    [50.1, 52.2, 56.2, 59.1, 49.6]
]).T  # shape = (5, 8)

# w/ adapters 数据
w_adapter = np.array([
    [67.6, 63.8, 69.8, 69.8, 74.2],
    [64.6, 59.9, 66.1, 71.0, 71.9],
    [85.8, 83.3, 87.3, 88.9, 92.0],
    [96.8, 97.9, 97.5, 98.1, 99.4],
    [76.9, 87.0, 86.7, 91.7, 97.1],
    [82.9, 87.0, 87.6, 96.5, 98.1],
    [97.8, 98.6, 98.5, 98.8, 99.6],
    [67.3, 69.4, 71.6, 73.6, 64.9]
]).T  # shape = (5, 8)

# 设置柱状图参数
x = np.arange(len(datasets))  # 每个数据集的位置
width = 0.15  # 每个柱子的宽度

# 为每个方法设置一对颜色 (w/o adapter, w/ adapter)
colors = ["#D79B00", "#B85450",
          "#5994C7", "#51B86C",
          "#8E44AD"]

fig, ax = plt.subplots(figsize=(14, 5))  # 拉长图宽

for i in range(len(methods)):
    # 下半部分（原始 w/o adapter）
    ax.bar(x + i*width, w_o_adapter[i], width,
           color=colors[i], edgecolor=colors[i], linewidth=1.3)
    
    # 上半部分（增幅，底色透明 + 斜线）
    ax.bar(x + i*width, w_adapter[i] - w_o_adapter[i], width,
           bottom=w_o_adapter[i],
           color=(0, 0, 0, 0),       # 底色透明
           edgecolor=colors[i],   # 斜线颜色
           hatch='///////',
           linewidth=1.3)

# X/Y 轴美化
ax.set_xticks(x + 2*width)
ax.set_xticklabels(datasets, rotation=0, fontsize=16)
ax.set_ylabel("Accuracy (%)", fontsize=16)
ax.set_ylim(0, 110)
ax.grid(axis="y", linestyle="--", alpha=0.5)

# --------------------
# 图例 1：方法（左上角）
legend1 = ax.legend(
    handles=[mpatches.Patch(color=colors[i], label=methods[i]) for i in range(len(methods))],
    loc='upper right', fontsize=12
)
ax.add_artist(legend1)  # 保留第一个图例

# 图例 2：w/o adapter 与 增幅（右上角）
legend2 = ax.legend(
    handles=[
        mpatches.Patch(color='grey', label='w/o adapter'),
        mpatches.Patch(facecolor='white', edgecolor='grey', hatch='///////', label='Increment (w/ adapter)')
    ],
    loc='upper left', fontsize=14
)
# --------------------

plt.tight_layout()
plt.savefig("./figures/exp_bar1.pdf", bbox_inches='tight', pad_inches=0)
plt.show()