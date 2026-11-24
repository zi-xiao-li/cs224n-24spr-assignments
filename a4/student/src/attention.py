
"""
最初从 Andrej Karpathy 的 minGPT 项目派生。

CS224N 2023-24：第四次作业

John Hewitt <johnhew@stanford.edu>
Ansh Khurana <anshk@stanford.edu>
Soumya Chatterjee <soumyac@stanford.edu>
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def precompute_rotary_emb(dim, max_positions):
    """
    RoPE 使用以下正弦和余弦函数来编码位置信息：

    cos(t theta_i) 和 sin(t theta_i)
        其中 t 表示位置，
              theta_i = 1/10000^(-2(i-1)/dim)，i ∈ [1, dim/2]

    由于序列的最大长度已知，我们可以提前计算这些值，
    以加快训练速度。

    请实现 precompute_rotary_emb 函数，返回一个形状为
    (max_positions, dim/2, 2) 的 Tensor，其中最后一个维度
    存储每个位置、每个 Embedding 维度对应的 cos 和 sin 值。
    """
    rope_cache = None
    # TODO: [part g]
    ### YOUR CODE HERE ###

    # 1. 计算每对特征对应的旋转频率 theta_i
    # theta 的形状：(dim/2,)
    theta = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

    # 2. 生成位置索引 t = 0, 1, ..., max_positions-1
    # positions 的形状：(max_positions,)
    positions = torch.arange(max_positions).float()

    # 3. 计算每个位置、每对特征的旋转角度 t * theta_i
    # angles 的形状：(max_positions, dim/2)
    angles = torch.outer(positions, theta)

    # 4. 计算余弦和正弦，并存储在最后一个维度
    # rope_cache 的形状：(max_positions, dim/2, 2)
    rope_cache = torch.stack(
        [torch.cos(angles), torch.sin(angles)],
        dim=-1
    )


    ### END YOUR CODE ###
    return rope_cache


def apply_rotary_emb(x, rope_cache):
    """将 RoPE 应用于输入张量 x。"""
    # TODO: [part g]
    # 以下函数可能有助于在实数与复数之间进行转换：

    # torch.view_as_real - https://pytorch.org/docs/stable/generated/torch.view_as_real.html
    # torch.view_as_complex - https://pytorch.org/docs/stable/generated/torch.view_as_complex.html

    # 注意：在推理过程中，输入序列的长度可能与预计算值的长度不同。
    # 在这种情况下，应截取预计算值，使其长度与输入序列长度一致。

    rotated_x = None
    ### YOUR CODE HERE ###

    # 1. 获取输入张量的形状
    B, nh, T, hs = x.shape

    # 2. 将相邻两个特征分为一组
    # (B, nh, T, hs) -> (B, nh, T, hs/2, 2)
    x_reshape = x.reshape(B, nh, T, hs // 2, 2)

    # 3. 将每对实数转换为一个复数
    # (B, nh, T, hs/2, 2) -> (B, nh, T, hs/2)
    x_complex = torch.view_as_complex(x_reshape.contiguous())

    # 4. 截取当前序列长度对应的旋转因子
    # (max_positions, hs/2, 2) -> (T, hs/2, 2)
    rope_cache = rope_cache[:T]

    # 5. 将 cos 和 sin 转换为复数：cos + i*sin
    # (T, hs/2, 2) -> (T, hs/2)
    rope_complex = torch.view_as_complex(rope_cache.contiguous())

    # 6. 通过复数乘法执行旋转
    # (B, nh, T, hs/2) * (T, hs/2)
    rotated_complex = x_complex * rope_complex

    # 7. 将复数转换回实数
    # (B, nh, T, hs/2) -> (B, nh, T, hs/2, 2)
    rotated_real = torch.view_as_real(rotated_complex)

    # 8. 恢复输入张量的原始形状
    # (B, nh, T, hs/2, 2) -> (B, nh, T, hs)
    rotated_x = rotated_real.reshape(B, nh, T, hs)


    ### END YOUR CODE ###
    return rotated_x



class CausalSelfAttention(nn.Module):
    """
    一个标准的多头掩码自注意力层，最后包含一个线性投影层。
    我本可以直接使用 torch.nn.MultiheadAttention，
    但它的文档几乎不存在，而且代码很难看，所以我不太信任它，
    因此在这里自行实现。
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # 为所有注意力头创建 Key、Query 和 Value 的线性投影层
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.rope = config.rope
        if self.rope:
            assert (config.n_embd % config.n_head) % 2 == 0 # 确保 Embedding 维度能被注意力头数整除，以便平均分配给每个 Attention Head

            # TODO: [part g] 预计算 RoPE 所需的 cos 和 sin 值，
            # 并将它们存储在 rope_cache 中。
            # 提示：最大序列长度由 config.block_size 给出。
            rope_cache = None
            ### YOUR CODE HERE ###

            # 每个 Attention Head 的特征维度
            head_dim = config.n_embd // config.n_head

            # 预计算 RoPE 的旋转因子
            rope_cache = precompute_rotary_emb(head_dim, config.block_size)

            ### END YOUR CODE ###

            self.register_buffer("rope_cache", rope_cache) # 将 rope_cache 注册为当前 PyTorch 模型的 Buffer（缓冲区张量）

        # 正则化
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # 输出投影层
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # 因果掩码：确保序列中每个位置只能关注自身及其左侧的位置
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()

        # 为 Batch 中所有注意力头计算 Query、Key 和 Value，
        # 并将注意力头维度提前，放在序列长度维度之前
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.rope:
            # TODO: [part g] 对 Query 和 Key 应用 RoPE。
            ### YOUR CODE HERE ###

            q = apply_rotary_emb(q, self.rope_cache)
            k = apply_rotary_emb(k, self.rope_cache)

            ### END YOUR CODE ###

        # 因果自注意力：(B, nh, T, hs) × (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) × (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 将所有注意力头的输出重新拼接在一起

        # 输出投影
        y = self.resid_drop(self.proj(y))
        return y


class CausalCrossAttention(nn.Module):
    """
    Modifications over the self-attention layer to handle two inputs and perform
    cross-attention between them.
    This follows the implementation of the self attention module with
    auto-regressive masking on (key).
    Manipulation of batch-size to allow for different batch size between the
    two inputs, with broadcasting over to the higher batch size value.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x_kv, x_q):
        Bk, Tk, Ck = x_kv.size()
        Bq, Tq, Cq = x_q.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim

        # keys of x1
        k = self.key(x_kv).view(Bk, Tk, self.n_head, Ck // self.n_head).transpose(1, 2) # (B, nh, Tk, hs)

        # query with x2
        q = self.query(x_q).view(Bq, Tq, self.n_head, Cq // self.n_head).transpose(1, 2) # (B, nh, Tq, hs)

        # values from x1
        v = self.value(x_kv).view(Bk, Tk, self.n_head, Ck // self.n_head).transpose(1, 2) # (B, nh, Tk, hs)

        # causal self-attention;  (B, nh, Tk, hs) x (B, nh, hs, Tq) -> (B, nh, Tq, Tk)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        B = max(Bk, Bq)

        att = att.masked_fill(self.mask[:,:,:Tq,:Tk] == 0, -1e10)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, Tq, Tk) x (B, nh, Tk, hs) -> (B, nh, Tq, hs)
        y = y.transpose(1, 2).contiguous().view(B, Tq, Cq) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
