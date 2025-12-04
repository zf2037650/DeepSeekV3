import torch
import torch.nn as nn

from MLA import MultiHeadLatentAttention as MLA
from MoE import V3MoE, SiluExperts
from torchtune.modules.rms_norm import RMSNorm

class DeepSeekV3Config:
    def __init__(
        self,
        vocab_size = 102400,
        dim = 2048,
        inter_dim = 10944,

        # moe
        n_activated_experts=6,
        n_expert_groups=8,
        n_limited_groups=4,
        route_scale=1,
        n_experts=256,
        n_shared_experts=2,
        hidden_dim=1408,

        # mla
        n_heads=16,
        q_lora_rank=512,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
        max_batch_size=8,
        max_seq_len=16384,
        n_layers = 4,
        n_dense_layers = 2,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.inter_dim = inter_dim

        self.topk_experts = n_activated_experts
        self.n_groups = n_expert_groups
        self.topk_groups = n_limited_groups
        self.route_scale = route_scale
        self.n_experts = n_experts
        self.n_shared_experts = n_shared_experts
        self.hidden_dim = hidden_dim

        self.n_heads = n_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_dense_layers = n_dense_layers

class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.n_layers = config.n_layers,
        self.n_dense_layers = config.n_dense_layers
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.attention = MLA(config)
        self.ffn =  SiluExperts(config.dim, config.inter_dim) if layer_id < config.n_dense_layers else V3MoE(config)

    def forward(self, x):
        x = self.ffn(self.norm2(x)) + x

        return x

class DeepSeekV3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeeding = nn.Linear(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.blocks.append(Block(config, layer_id))
        self.out_norm = RMSNorm(config.dim)
        self.out = nn.Linear(config.dim, config.vocab_size)

    def forward(self, x):
        x = self.embeeding(x)
        for block in self.blocks:
            x = block(x)
        out = self.out(self.out_norm(x))

        return out

if __name__ == "__main__":
    config = DeepSeekV3Config()
    torch.set_default_dtype(torch.bfloat16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dpv3 = DeepSeekV3(config).to(device)
    input = torch.randint(1, (8, 8, 102400)).to(dtype=torch.bfloat16)
    output = dpv3(input)
