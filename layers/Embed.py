import torch
from torch import nn
import torch.nn.functional as F
import math

# ==================== 基础模块 ====================

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [B, seq_len, d_model]
        Returns:
            [B, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DataEmbedding(nn.Module):
    """
    数据嵌入模块
    将输入特征映射到d_model维度，并添加位置编码
    """
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEncoding(d_model, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        """
        Args:
            x: [B, seq_len, c_in]
        Returns:
            [B, seq_len, d_model]
        """
        x = self.value_embedding(x)
        x = self.position_embedding(x)
        return self.dropout(x)

