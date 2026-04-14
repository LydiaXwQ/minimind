from transformers import PretrainedConfig

class MiniMindConfig(PretrainedConfig):
    model_type = "MiniMind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn
import math
from transformers.activations import ACT2FN
from typing import Optional, Tuple
import torch.nn.functional as F

# inherit nn.Module
class RMSNorm(nn.Module):

# Initialization method
    def __init__(self, dim:int, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

# _norm
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

# forward
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_freqs_cis(dim: int, end:int = int(32*1024), rope_base: float = 1e6, rope_scaling: dict = None):
    # 1. 初始化RoPE频率：freqs = 1/(ropebase^(2i/d))
    # i = 0, 1, 2, ..., d/2-1 since we need d/2 pairs of dimenions
    # so 2i = 0, 2, 4, ..., d-2, since arange excludes the end value, we can use arange(0, dim, 2) to get the even indices
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end / orig_max > 1.0:
            # 找到i：i = dln(L/(b*2pi))/2ln(base)
            # 定义一个接收b的函数：
            inv_dim = lambda b : (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            # 调用inv_dim 算出i_fast和i_slow，需要注意i范围是[0, d/2-1]
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)

            #计算γ -> ramp：
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡: (i - low) / (high - low)
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp((torch.arange(0, dim//2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)

            # 对frequency进行缩放：
            # 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            freqs = freqs * ((1.0 - ramp) + (ramp / factor))

    # 根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device=freqs.device)

    # 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    # shape：(end, dim/2)
    freqs = torch.outer(t, freqs).float()

    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        # 将输入张量 x 沿最后一个维度分成两半，并交换这两半的位置
        # [a, b, c, d] -> [-c, -d, a, b]
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复key-value张量以匹配query头数 (用于分组查询注意力GQA)
    等价于torch.repeat_interleave(x, dim=2, repeats=n_rep)，但更高效
    
    在GQA中，key和value的头数少于query，需要重复来匹配
    例如：8个query头，2个kv头，则需要每个kv头重复4次
    
    Args:
        x: kv张量 [batch, seq_len, num_kv_heads, head_dim]
        n_rep: 重复次数
    
    Returns:
        重复后的张量 [batch, seq_len, num_kv_heads * n_rep, head_dim]
    """

    # KV 数 = Q数 （等于MHA情况），直接返回
    if n_rep == 1:
        return x
    bs, slen, n_kv, head_dim = x.shape
    #[bs, slen, n_kv, 1, head_dim]
    x_addDim = x[:, :, :, None, :]
    # [bs, slen, n_kv, n_rep, head_dim]
    x_rep = x_addDim.expand(bs, slen, n_kv, n_rep, head_dim)
    # [bs, slen, n_kv*n_rep, head_dim]
    result = x_rep.reshape(bs, slen, n_kv * n_rep, head_dim)

    return result

class Attention(nn.Module):
    """
    多头自注意力机制，支持分组查询注意力(GQA)和Flash Attention优化
    
    GQA介绍：
    - 传统MHA：query、key、value头数相同
    - GQA：key、value头数少于query头数，通过重复匹配
    - 优点：减少KV cache内存占用，保持性能
    """
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        # 检查Q的数量必须被KV的数量整除
        assert args.num_attention_heads % self.num_key_value_heads == 0
        
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // self.n_local_heads

        self.q_proj = nn.Linear(args.hidden_size, self.n_local_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(self.n_local_heads * self.head_dim, args.hidden_size, bias=False)
        
        # Dropout层用于正则化
        self.attn_dropout = nn.Dropout(args.dropout)    # 注意力权重dropout
        self.resid_dropout = nn.Dropout(args.dropout)   # 残差连接dropout
        self.dropout = args.dropout                      # 保存dropout率

        # 检查是否支持Flash Attention
        # hasattr(obj, 'attr'): 检查对象是否有指定属性
        # Flash Attention需要PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attention
        # 如果不支持可以打印警告: print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                    x: torch.Tensor,
                    position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    use_cache=False,
                    attention_mask: Optional[torch.Tensor] = None):
    
        # 投影， 计算q，k，v
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # 把输入拆分成多个头， 用view
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # q和k，使用RoPE
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len] , sin[:seq_len] )

        # -------------------- KV cache 处理 --------------------
        # 如果past_key_value不为None，说明是生成阶段，需要把当前的k和v与缓存的k和v拼接起来
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        past_kv = (xk, xv) if use_cache else None

        # 对于k和v使用repeat，补齐KV和Q相差的维度 （注意kv cache）
        # i.e Q:[head0, head1, head2, head3], K,V:[head0, head1], 需要把K,V重复成[head0, head0, head1, head1]，才能和Q进行点积
        xq = xq.transpose(1, 2)  # [bsz, seq_len, self.n_local_heads, self.head_dim] -> [bsz, self.n_local_heads, seq_len, self.head_dim]
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)  # [bsz, seq_len, self.n_local_kv_heads * n_rep, self.head_dim] -> [bsz, self.n_local_kv_heads * n_rep, seq_len, self.head_dim]
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # -------------------- Attention计算 --------------------
        # 优先使用PyTorch 2.0+的scaled_dot_product_attention（Flash Attention实现）
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # 如果没有显式的attention_mask，直接传None让底层高效实现
            attn_mask = None if attention_mask is None else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool()
            # F.scaled_dot_product_attention是PyTorch在新版本中提供的高效实现
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True  # 自回归（因果）注意力
            )
        else:
            # 进行attention计算，q@k^T / sqrt(d)，如果有mask就加上mask，最后softmax
            # [bsz, self.n_local_heads, seq_len, self.head_dim] @ [bsz, self.n_local_kv_heads * n_rep, self.head_dim, seq_len]
            # -> [bsz, self.n_local_heads, seq_len, seq_len]
            scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_dim)
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
            # 如果有attention_mask(0/1)，将其扩展后转为 -1e9 的加性mask（掩掉pad位置）
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
            
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            # 最后拼接头，输出投影，返回
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # 恢复形状并做输出投影 + 残差dropout
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  # [bsz, seq_len, hidden]
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    # 初始化
    def __init__(self, args:MiniMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size*8/3)
            args.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        # 升维
        # up_proj: hidden -> intermediate (用于被gate的部分)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)

        # SwiGLU类似于Gated Linear Unit变体：act(gate(x)) * up(x)

        # 降维
        # down_proj: intermediate -> hidden (用于投影回hidden维度)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        # 门控
        # gate_proj: hidden -> intermediate (用于计算gate部分)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        # dropout
        self.dropout = nn.Dropout(args.dropout)
        # 激活函数
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))
