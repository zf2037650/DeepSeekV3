import math

import torch
import torch.nn as nn
from torchtune.modules.rms_norm import RMSNorm
from torchtune.modules import RotaryPositionalEmbeddings
import torch.nn.functional as F

class MLAConfig:
    def __init__(
            self,
            dim = 2048,
            n_heads = 16,
            q_lora_rank = 512 ,
            kv_lora_rank = 512,
            qk_nope_head_dim = 128,
            qk_rope_head_dim = 64,
            qk_head_dim = 192,
            v_head_dim = 128,
            max_batch_size = 8,
            max_seq_len = 16384

    ):
        self.dim = dim
        self.n_heads = n_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.kv_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.rope = RotaryPositionalEmbeddings(self.qk_rope_head_dim)

        # wkv_a: 将输入投影到低维潜在空间 + 位置编码部分
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)

        # wkv_b: 从低维潜在空间恢复出完整的K和V
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)

        # Softmax缩放因子
        self.softmax_scale = self.qk_head_dim ** -0.5

        # 只缓存低维的潜在表示，形状为 [batch, seq_len, kv_lora_rank]
        self.register_buffer("kv_cache", torch.zeros(config.max_batch_size, config.max_seq_len, self.kv_lora_rank), persistent=False)
        self.register_buffer("pe_cache", torch.zeros(config.max_batch_size, config.max_seq_len, self.qk_rope_head_dim), persistent=False)
        self.cache_pos = 0


    def forward(self, x, start_pos=None, freqs_cis=None, mask=None):

        batch_size, seq_len, _ = x.shape

        if start_pos is None:
            start_pos = self.cache_pos

        end_pos = start_pos + seq_len

        # 获取q
        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(batch_size, seq_len, self.n_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_rope_pos = self.rope(q_rope)
        q = torch.cat([q_nope, q_rope_pos], dim=-1)

        # 获取kv
        kv_a = self.wkv_a(x)
        kv_latent, k_pe = torch.split(kv_a,[self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        #k_pe = k_pe.view(batch_size, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        k_pe_pos = self.rope(k_pe.unsqueeze(2))

        # 缓存保存的是映射后的低秩结果
        self.kv_cache[:batch_size, start_pos:end_pos] = kv_latent
        self.pe_cache[:batch_size, start_pos:end_pos] = k_pe_pos.view(batch_size, seq_len, -1)
        self.cache_pos = end_pos

        cached_kv = self.kv_cache[:batch_size, :end_pos]
        cached_pe = self.pe_cache[:batch_size, :end_pos]

        k_rope = cached_pe.unsqueeze(1)
        # 做expand，不然没法concat
        k_rope = k_rope.expand(-1, self.n_heads, -1, -1)
        kv = self.wkv_b(self.kv_norm(cached_kv)).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # 得到q,k,v后进行MHA的编写
        attn_weights = torch.matmul(
            q, k.transpose(2, 3)
        )
        attn_weights = attn_weights / math.sqrt(self.qk_head_dim)
        if mask is not None:
            attn_weights = torch.masked_fill(
                attn_weights,
                mask == 0,
                float("-inf")
            )

        attn_weights = F.dropout(F.softmax(attn_weights, dim=-1).to(q.dtype))
        output = torch.matmul(attn_weights, v)
        output = self.wo(output.transpose(1, 2).reshape(batch_size, seq_len, -1))
        return output



if __name__ == "__main__":
    config = MLAConfig()
    torch.set_default_dtype(torch.bfloat16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    moe = MultiHeadLatentAttention(config).to(device)
    input = torch.randint(1, (8, 8, 2048)).to(dtype=torch.bfloat16)
    output = moe(input)