import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers import (
    GPT2ForSequenceClassification,
    GPT2Model,
    GPT2Tokenizer,
    default_data_collator,
    AutoConfig
)


class AdapterLayer(nn.Module):
    """
    Adapter层实现
    将其添加到Transformer层中可以实现参数高效微调
    """
    def __init__(self, input_size, adapter_size):
        super(AdapterLayer, self).__init__()
        # 降维全连接层
        self.down_project = nn.Linear(input_size, adapter_size)
        # 激活函数
        self.activation = nn.ReLU()
        # 升维全连接层
        self.up_project = nn.Linear(adapter_size, input_size)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        # 初始化down_project用较小的值
        nn.init.normal_(self.down_project.weight, std=1e-3)
        # nn.init.xavier_uniform_(self.down_project.weight)
        # nn.init.zeros_(self.down_project.weight)
        nn.init.zeros_(self.down_project.bias)
        
        # 初始化up_project为接近零的值，确保训练初期对原始模型影响较小
        nn.init.normal_(self.up_project.weight, std=1e-3)
        # nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x):
        # 保存原始输入用于残差连接
        residual = x
        
        # 通过降维层
        x = self.down_project(x)
        # 激活
        x = self.activation(x)
        # 通过升维层
        x = self.up_project(x)
        
        # 残差连接
        return residual - x 


class GPT2BlockWithAdapter(nn.Module):
    """
    带Adapter的GPT2Block层
    在原始GPT2Block的基础上添加Adapter层实现参数高效微调
    """
    def __init__(self, config):
        super(GPT2BlockWithAdapter, self).__init__()
        # 创建标准的GPT2Block
        self.original_block = GPT2Block(config)
        
        # 添加Adapter层
        adapter_size = 64  # Adapter的隐藏层大小
        self.adapter = AdapterLayer(config.hidden_size, adapter_size)
    
    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs  # 使用**kwargs接收所有其他参数
    ):
        # 首先通过原始的GPT2Block，只传递它支持的参数
        outputs = self.original_block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        
        # 原始输出中的第一个元素是隐藏状态
        hidden_states = outputs[0]
        
        # 将隐藏状态通过Adapter层
        hidden_states = self.adapter(hidden_states)
        
        # 更新输出的隐藏状态
        outputs = (hidden_states,) + outputs[1:]
        
        return outputs
    
    def load_state_dict(self, state_dict, strict=True):
        """
        自定义加载参数方法，用于从原始GPT2Block加载参数
        """
        # 将所有参数传递给原始Block
        return self.original_block.load_state_dict(state_dict, strict=strict) 



class GPT2ClassifierWithAdapter(nn.Module):
    def __init__(self, pretrained_model_name):
        super(GPT2ClassifierWithAdapter, self).__init__()
        # 加载预训练模型
        self.gpt2 = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=pretrained_model_name)
        
        # 确保模型配置中设置了pad_token_id
        # self.gpt2.config.pad_token_id = self.gpt2.config.eos_token_id
        self.gpt2.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name)
        
        # 替换原始的GPT2Block为带Adapter的版本
        config = self.gpt2.config
        for i in range(len(self.gpt2.transformer.h)):
            # 保存原始权重
            old_block = self.gpt2.transformer.h[i]
            # 创建带Adapter的新Block
            new_block = GPT2BlockWithAdapter(config)
            # 复制原始权重
            new_block.load_state_dict(old_block.state_dict(), strict=False)
            # 替换
            self.gpt2.transformer.h[i] = new_block
            
        # 冻结原始GPT2参数
        for param in self.gpt2.parameters():
            param.requires_grad = False
            
        # 解冻分类器层和Adapter层参数
        # for param in self.gpt2.score.parameters():
        #     param.requires_grad = True
            
        # 解冻所有Adapter层
        for i in range(len(self.gpt2.transformer.h)):
            for param in self.gpt2.transformer.h[i].adapter.parameters():
                param.requires_grad = True
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )