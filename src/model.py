import math
import torch
import torch.nn as nn
from src.attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model precisa ser perfeitamente divisível por n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        
        self.w_o = nn.Linear(d_model, d_model)
        
        
        self.attention = ScaledDotProductAttention(dropout=dropout)
        
    def split_heads(self, tensor: torch.Tensor):
        
        batch_size, seq_len, _ = tensor.size()
        return tensor.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        batch_size = q.size(0)
        
        
        q = self.split_heads(self.w_q(q))
        k = self.split_heads(self.w_k(k))
        v = self.split_heads(self.w_v(v))
        
        
        out, attention_weights = self.attention(q, k, v, mask=mask)
        
        
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        
        return self.w_o(out), attention_weights


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, num_classes: int = 1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        
        self.attention = MultiHeadAttention(d_model, n_heads)
        
        
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x: torch.Tensor):
        
        out = self.embedding(x)  
        
        
        out, weights = self.attention(out, out, out)
        
        
        out = out.mean(dim=1)  
        
    
        return self.fc(out), weights


# =====================================================================
# BLOC DE TESTE DE SANIDADE (Opcional para verificação local de integridade)
# =====================================================================
if __name__ == "__main__":
    
    simulated_batch = torch.randint(0, 100, (2, 10))  
    
    classifier = TransformerClassifier(vocab_size=100, d_model=64, n_heads=4)
    logits, attn_maps = classifier(simulated_batch)
    
    print("=== INTEGRIDADE DO CLASSIFICADOR TRANSFORMER ===")
    print(f"Shape dos Logits de Saída (Deve ser [2, 1]): {logits.shape}")
    print(f"Shape do Mapa de Atenção (Deve ser [2, 4, 10, 10]): {attn_maps.shape}")