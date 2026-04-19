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
from typing import Optional, Tuple, Union, List
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel, GenerationMixin
from torch.nn import init

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

class MinimindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):

        # 残差连接模式：先做LayerNorm -> Attention -> 残差相加 -> LayerNorm -> FFN -> 残差相加
        # 保存残差以供后续相加
        residual = hidden_states
        # 注意力子层：输入先归一化（RMSNorm），返回hidden_states和present_key_value（用于cache）
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),  # 输入先做LayerNorm
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )

        hidden_states = residual + hidden_states  # 注意力输出与残差相加
        residual = hidden_states  # 更新残差

        # FFN子层：输入先归一化（RMSNorm），返回hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))  # 输入先做LayerNorm
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MinimindBlock(i, config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE预计算
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim = config.hidden_size // config.num_attention_heads,
            end = config.max_position_embeddings,
            rope_base = config.rope_theta,
            rope_scaling = config.rope_scaling
        )

        # 将预计算的频率注册为buffer，这样它们就会随着模型一起移动到GPU/TPU，并且不会被优化器更新
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        
    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs):

        batch_size, seq_len = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None
        
        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算start_pos：如果存在past，则start_pos为已有past序列长度
        # past_key_values[0] 形如 (k, v)，k.shape = [bsz, past_seq_len, n_kv_heads, head_dim]
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # Embedding + dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 从注册的buffer中取出对应位置范围的cos/sin作为position_embeddings
        # self.freqs_cos/freqs_sin的shape为 [max_pos, head_dim]
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_len],
            self.freqs_sin[start_pos:start_pos + seq_len]
        )

        # 逐层前向，通过zip把layer和对应的past_key_value配对
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 最后做归一化
        hidden_states = self.norm(hidden_states)

        return hidden_states, presents

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    def __init__(self, config: MiniMindConfig):
        super().__init__(config)
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 权重共享
        # 输出层的权重和潜入层的权重共享
        self.model.embed_tokens.weight = self.lm_head.weight

    
    def forward(self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    use_cache: bool = False,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **args,):
    
        hidden_states, presents = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )

        # logits to keep 是整数，保留最后n个位置：
        # 生成的时候只需要最后的logits来预测下一个token
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )

        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )


        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=hidden_states
        )

class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim) # shape = (bsz*seq_len, hidden_dim)
        # 计算出每个token对各个expert的logits
        logits = F.linear(hidden_states, self.weight, None)

        # Sb,e: 每个token对每个专家的分数，shape = (bsz*seq_len, n_routed_experts)
        if self.scoring_func == "softmax":
            scores = F.softmax(logits, dim=-1)
        else:
            raise NotImplementedError(f"Scoring function {self.scoring_func} not implemented")
    
        # 两种aux loss计算方式：
        # 序列级别
        # Batch级别
        topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            # 对每个token的topk权重求和
            denominator = (
                topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            )
            # 每个token的topk权重除以这个和，得到归一化的权重
            topk_weight = topk_weight / denominator

        # 计算辅助损失（仅训练时）
        # 辅助损失的作用：确保均衡负载，防止所有token都流向少数专家

        # 判断是在训练模式下且alpha > 0才计算aux loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # 将topk从[bsz*seq_len, top_k]的索引转换回[bsz, seq_len, top_k]的形式，以便后续计算
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)

            # ---- 方式1: 序列级辅助损失（seq_aux=True) ----
            if self.seq_aux:
                # 恢复scores维度：
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # 统计每个batch中每个专家被选中的次数：
                # ce: [bsz, n_routed_experts]，记录每个专家被选中的次数
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )

                # scatter_add_: 根据topk_idx_for_aux_loss中的索引，将1累加到ce的对应位置上，统计每个专家被选中的次数
                # 例如：如果topk_idx_for_aux_loss[0] = [1, 3]，则ce[0, 1]和ce[0, 3]都会加1，表示第0个batch中第1和第3个专家被选中了一次
                ce.scatter_add_(
                    dim=1,
                    index=topk_idx_for_aux_loss,
                    src=torch.ones_like(topk_idx_for_aux_loss, dtype=ce.dtype)
                )

                # 计算每个专家的平均使用率：
                # ce/总的选择数，得到每个专家被选中的平均概率
                # ce / L * K / E，其中L是序列长度，K是每个token选择的专家数，E是总的路由专家数
                ce = ce.div(
                    seq_len * aux_topk / self.n_routed_experts
                )

                # 计算aux loss：相对负载率 × 平均概率，求和后乘以 alpha
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1, keepdim=True)).sum(
                    dim=1
                ).mean() * self.alpha

            # ---- 方式2: Batch级辅助损失（seq_aux=False）----
            else:
                # 将topk_idk 展平：[bsz, seq_len, top_k] -> [bsz*seq_len*top_k]
                # 转换为one_hot编码， 得到[bsz*seq_len*top_k, n_routed_experts]，每行只有一个1，表示对应的专家被选中
                # 举个例子：如果n_routed_experts=5，topk_idx_for_aux_loss 包含值0-4
                # 那么one_hot编码会将这些值转化为对应位置为1，其余位置为0的向量。
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                
                # ce表示整个batch中每个专家的平均选择率
                # 比如[0.2, 0.5, 0.3]表示第一个专家被选中的平均概率是20%，第二个是50%，第三个是30% 
                # fe = sum(mi,e) / [1/(N*K)]
                ce = mask_ce.float().mean(dim=0)
                # 乘以 E 把占比变回相对负载率：
                fi = ce * self.n_routed_experts
                # Pi = 1/N * Sum（Si，e）
                # 表示整个batch中每个专家的平均分数
                # scores 不恢复维度： [bsz * seq_len, n_routed_experts]
                Pi = scores_for_aux.mean(dim=0)
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss




                









