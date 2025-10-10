import matplotlib.pyplot as plt

# 原始数据
innersteps = [0, 1, 5, 10]
avg_acc = [0.7489452276911054, 0.7642297233854022, 0.7573949524334499, 0.7579045976911273]
time_cost = [4465.492212533951, 7291.446763753891, 8736.942529201508, 10413.920747995377]

# 映射到等间隔位置 [0,1,2,3]
x_positions = list(range(len(innersteps)))

fig, ax1 = plt.subplots(figsize=(5.5, 3.5))

# 左轴：准确率
color = "tab:blue"
ax1.set_xlabel("inner step", fontsize=18)
ax1.set_ylabel("Average Accuracy", color=color, fontsize=18)
ax1.plot(x_positions, avg_acc, marker="o", color=color, label="Average Accuracy")
ax1.tick_params(axis="y", labelcolor=color)
ax1.set_ylim(0.7, 0.78)

# 设置等间隔刻度 + 标签
ax1.set_xticks(x_positions)
ax1.set_xticklabels(innersteps)

# 右轴：时间
ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Time Cost (s)", color=color, fontsize=18)
ax2.plot(x_positions, time_cost, marker="s", linestyle="--", color=color, label="Time Cost")
ax2.tick_params(axis="y", labelcolor=color)

# 图例合并
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="lower right", fontsize=15, frameon=True, shadow=True, edgecolor="gray")

# plt.title("Average Accuracy and Time vs. Innerstep", fontsize=17)
plt.tight_layout()
plt.savefig("./figures/linechart_GPT.pdf", bbox_inches='tight', pad_inches=0)