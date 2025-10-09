import re
import matplotlib.pyplot as plt
from collections import defaultdict

old_file = "/home/cj/EMR_Merging/merge_vit/logs/ViT-B-32/epoch3000_inner1_0.01_0.1.txt"
new_file = "/home/cj/EMR_Merging/merge_vit/logs/ViT-B-32/log_2025_09_20_15_56_59_1_0.01_0.1_weightave_adapter_16.txt"

def parse_adapter_losses(filepath):
    dataset_losses = defaultdict(list)
    current_dataset = None
    with open(filepath, "r") as f:
        for line in f:
            # 找数据集开始
            start_match = re.search(r"Start to train adapter for (\w+)", line)
            if start_match:
                current_dataset = start_match.group(1)
                continue
            # 找 loss 行
            loss_match = re.search(r"loss ([0-9.eE+-]+)", line)
            if loss_match and current_dataset is not None:
                dataset_losses[current_dataset].append(float(loss_match.group(1)))
    return dataset_losses

# 解析新旧日志
old_adapters = parse_adapter_losses(old_file)
new_adapters = parse_adapter_losses(new_file)

# 画图
for dataset in new_adapters.keys():
    plt.figure(figsize=(4,3))
    if dataset in old_adapters:
        plt.plot(range(len(old_adapters[dataset])), old_adapters[dataset], label="MetaMerging", color='blue')
        # print(len(old_adapters[dataset]))
    plt.plot(range(len(new_adapters[dataset])), new_adapters[dataset], label="Weight Averaging", color='gold')
    
    plt.xlabel("Training Step", fontsize=14, labelpad=0)
    plt.ylabel("Loss", fontsize=14, labelpad=0)
    plt.title(f"{dataset}", fontsize=16)
    plt.legend(fontsize=13, frameon=True, shadow=True, edgecolor="gray", loc='upper right')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"./figures/loss_{dataset}.pdf", bbox_inches='tight', pad_inches=0)