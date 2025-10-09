from huggingface_hub import snapshot_download

# 例如：下载整个模型 repo
snapshot_download(
    repo_id="tanganke/gpt2_mrpc",   # 模型或数据集名称
    # repo_id='openai-community/gpt2',
    local_dir="/HDDDATA/data_cj/ckpts/gpt2/gpt2_mrpc",       # 存储路径
    local_dir_use_symlinks=False   # 避免使用软连接（适合非 Unix）
)
snapshot_download(
    repo_id="tanganke/gpt2_qnli",   # 模型或数据集名称
    # repo_id='openai-community/gpt2',
    local_dir="/HDDDATA/data_cj/ckpts/gpt2/gpt2_qnli",       # 存储路径
    local_dir_use_symlinks=False   # 避免使用软连接（适合非 Unix）
)
snapshot_download(
    repo_id="tanganke/gpt2_qqp",   # 模型或数据集名称
    # repo_id='openai-community/gpt2',
    local_dir="/HDDDATA/data_cj/ckpts/gpt2/gpt2_qqp",       # 存储路径
    local_dir_use_symlinks=False   # 避免使用软连接（适合非 Unix）
)
snapshot_download(
    repo_id="tanganke/gpt2_rte",   # 模型或数据集名称
    # repo_id='openai-community/gpt2',
    local_dir="/HDDDATA/data_cj/ckpts/gpt2/gpt2_rte",       # 存储路径
    local_dir_use_symlinks=False   # 避免使用软连接（适合非 Unix）
)
snapshot_download(
    repo_id="tanganke/gpt2_sst2",   # 模型或数据集名称
    # repo_id='openai-community/gpt2',
    local_dir="/HDDDATA/data_cj/ckpts/gpt2/gpt2_sst2",       # 存储路径
    local_dir_use_symlinks=False   # 避免使用软连接（适合非 Unix）
)
