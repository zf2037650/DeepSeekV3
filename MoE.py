import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEConfig:
    def __init__(
            self,
            dim = 2048,
            n_activated_experts = 6,
            n_expert_groups =  8,
            n_limited_groups = 4,
            route_scale = 1,
            n_experts = 256,
            n_shared_experts = 2,
            hidden_dim = 1408,
    ):
        self.dim = dim
        self.topk_experts = n_activated_experts
        self.n_groups = n_expert_groups
        self.topk_groups = n_limited_groups
        self.route_scale = route_scale
        self.n_experts = n_experts
        self.n_shared_experts = n_shared_experts
        self.hidden_dim = hidden_dim

class V3MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.topk_experts = config.topk_experts
        self.n_groups = config.n_groups
        self.topk_groups = config.topk_groups
        self.route_scale = config.route_scale
        self.n_experts = config.n_experts

        self.gate = nn.Linear(self.dim, self.n_experts)
        self.bias = nn.Parameter(torch.zeros(self.n_experts))
        self.experts = nn.ModuleList([SiluExperts(config.dim, config.hidden_dim) if i < self.n_experts else None
                                      for i in range(self.n_experts)])
        self.shared_experts = SiluExperts(config.dim, config.hidden_dim)

    def forward(self, x):
        # 首先编写门控部分的网络结构，获取选取的专家索引与权重
        original_shape = x.size()
        x = x.view(-1, self.dim)
        # 添加偏置进行负载均衡
        scores = F.softmax(self.gate(x), dim=-1)
        # 核对发现参考代码中用的软拷贝，参考代码有误？
        original_scores = scores.detach().clone()
        scores += self.bias.view(1, -1)
        scores = scores.view(x.size(0), self.n_groups, -1)
        group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
        index = group_scores.topk(self.topk_groups, dim=-1)[1]
        mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, index, False)
        scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        weight, index = torch.topk(scores, self.topk_experts, dim=-1)
        weight = original_scores.gather(1, index) * self.route_scale

        # 进行输出部分代码的编写
        y = torch.zeros_like(x)
        counts = torch.bincount(index.flatten(), minlength=self.n_experts).tolist()
        for i in range(self.n_experts):
            if counts[i] == 0:
                continue
            token_raw, group_numbers = torch.where(index==i)
            y[token_raw] += self.experts[i](x[token_raw]) * weight[token_raw, group_numbers, None]
        return (y + self.shared_experts(x)).view(original_shape)


class SiluExperts(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.upLinear = nn.Linear(dim1, dim2)
        self.downLinear = nn.Linear(dim2, dim1)
        self.gateLinear = nn.Linear(dim1, dim2)

    def forward(self, x):
        return self.downLinear(F.silu(self.upLinear(x)) * self.gateLinear(x))


if __name__ == "__main__":
    config = MoEConfig()
    torch.set_default_dtype(torch.bfloat16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    moe = V3MoE(config).to(device)
    input = torch.randint(1, (8, 8, 2048)).to(dtype=torch.bfloat16)
    output = moe(input)










