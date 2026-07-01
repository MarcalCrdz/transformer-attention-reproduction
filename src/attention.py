import math
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        d_k = q.size(-1)
        
        # Q K^T
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = self.softmax(scores)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights