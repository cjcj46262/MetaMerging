from huggingface_hub import snapshot_download



snapshot_download(
    repo_id='openai-community/gpt2',      
    local_dir="/HDDDATA/data/ckpts/gpt2/gpt2",     
    local_dir_use_symlinks=False   
)
snapshot_download(
    repo_id="tanganke/gpt2_cola",   
    # repo_id='openai-community/gpt2',      
    local_dir="/HDDDATA/data/ckpts/gpt2/gpt2_cola",     
    local_dir_use_symlinks=False   
)
snapshot_download(
    repo_id="tanganke/gpt2_mnli",   
    # repo_id='openai-community/gpt2',
    local_dir="/HDDDATA/data/ckpts/gpt2/gpt2_mnli",     
    local_dir_use_symlinks=False   
)
snapshot_download(
    repo_id="tanganke/gpt2_mrpc",   
    # repo_id='openai-community/gpt2',
    local_dir="/HDDDATA/data/ckpts/gpt2/gpt2_mrpc",       
    local_dir_use_symlinks=False   
)
snapshot_download(
    repo_id="tanganke/gpt2_qnli",  
    # repo_id='openai-community/gpt2',
    local_dir="/HDDDATA/data/ckpts/gpt2/gpt2_qnli",    
    local_dir_use_symlinks=False   
)
snapshot_download(
    repo_id="tanganke/gpt2_qqp",   
    # repo_id='openai-community/gpt2',
    local_dir="/HDDDATA/data/ckpts/gpt2/gpt2_qqp",      
    local_dir_use_symlinks=False   
)
snapshot_download(
    repo_id="tanganke/gpt2_rte",   
    # repo_id='openai-community/gpt2',
    local_dir="/HDDDATA/data/ckpts/gpt2/gpt2_rte",      
    local_dir_use_symlinks=False 
)
snapshot_download(
    repo_id="tanganke/gpt2_sst2",   
    # repo_id='openai-community/gpt2',
    local_dir="/HDDDATA/data/ckpts/gpt2/gpt2_sst2",     
    local_dir_use_symlinks=False   
)
