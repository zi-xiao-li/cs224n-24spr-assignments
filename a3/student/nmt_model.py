
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2024 春季：作业 3
nmt_model.py：神经机器翻译（NMT）模型
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
Siyan Li <siyanli@stanford.edu>
Moussa Doumbouya <moussa@stanford.edu>
"""

from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ModelEmbeddings

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):
    """ 简单的神经机器翻译模型：
        - 双向 LSTM 编码器
        - 单向 LSTM 解码器
        - 全局注意力模型（Luong 等，2015）
    """

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        """ 初始化 NMT 模型。

        @param embed_size (int): 词嵌入大小（维度）
        @param hidden_size (int): 隐藏状态大小（维度）
        @param vocab (Vocab): 包含源语言和目标语言词表的词表对象
                              具体说明参见 vocab.py。
        @param dropout_rate (float): 用于注意力机制的 Dropout 概率
        """
        super(NMT, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # 默认值
        self.post_embed_cnn = None
        self.encoder = None
        self.decoder = None
        self.h_projection = None
        self.c_projection = None
        self.att_projection = None
        self.combined_output_projection = None
        self.target_vocab_projection = None
        self.dropout = None

        # 仅用于合理性检查，与具体实现无关
        self.gen_sanity_check = False
        self.counter = 0

        ### 在此处编写代码（约 9 行）
        ### TODO - 严格按照以下顺序初始化这些变量：
        ###     self.post_embed_cnn（Conv1d 层，卷积核大小为 2，输入和输出通道数均为 embed_size，
        ###         padding = 'same'，以保持输出形状不变）
        ###     self.encoder（带偏置项的双向 LSTM）
        ###     self.decoder（带偏置项的 LSTMCell）
        ###     self.h_projection（无偏置项的线性层），在 PDF 中记为 W_{h}。
        ###     self.c_projection（无偏置项的线性层），在 PDF 中记为 W_{c}。
        ###     self.att_projection（无偏置项的线性层），在 PDF 中记为 W_{attProj}。
        ###     self.combined_output_projection（无偏置项的线性层），在 PDF 中记为 W_{u}。
        ###     self.target_vocab_projection（无偏置项的线性层），在 PDF 中记为 W_{vocab}。
        ###     self.dropout（Dropout 层）
        ###
        ### 请参考以下文档，正确初始化这些变量：
        ###     Conv1d：
        ###         https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        ###     LSTM：
        ###         https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        ###     LSTMCell：
        ###         https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
        ###     Linear 层：
        ###         https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        ###     Dropout 层：
        ###         https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html

        # 1. 一维卷积：输入和输出通道均为 embed_size，保持序列长度不变
        self.post_embed_cnn = nn.Conv1d(embed_size, embed_size, 2, padding="same")
        # 2. 双向 LSTM Encoder：每个方向的隐藏状态维度为 hidden_size
        self.encoder = nn.LSTM(embed_size, hidden_size, bias=True ,bidirectional=True)
        # 3. LSTMCell Decoder：输入为词嵌入与上一时刻输出向量的拼接
        self.decoder = nn.LSTMCell(embed_size + hidden_size, hidden_size, bias=True)
        #
        self.h_projection = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.c_projection = nn.Linear(2*hidden_size, hidden_size, bias=False)
        # nn.Linear(in_features, out_features) 的权重矩阵形状是 (out_features, in_features)
        self.att_projection = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.combined_output_projection = nn.Linear(3*hidden_size, hidden_size, bias=False)
        # 为目标词表中的每一个 token 输出一个分数:out_features
        self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.tgt), bias=False)
        # Dropout层
        self.dropout = nn.Dropout(p=dropout_rate)


        ### 代码结束


    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)  # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)  # Tensor: (tgt_len, b)

        ###     Run the network forward:
        ###     1. Apply the encoder to `source_padded` by calling `self.encode()`
        ###     2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        ###     3. Apply the decoder to compute combined-output by calling `self.decode()`
        ###     4. Compute log probability distribution over the target vocabulary using the
        ###        combined_outputs returned by the `self.decode()` function.

        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores


    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ 将编码器应用于源语言句子，得到编码器的隐藏状态。
            此外，将编码器的最终状态进行投影，得到解码器的初始状态。

        @param source_padded (Tensor): 填充后的源语言句子张量，形状为 (src_len, b)，其中
                                      b = batch_size，src_len = 源语言句子的最大长度。
                                      注意，这些句子已经按照长度从长到短排序。
        @param source_lengths (List[int]): 当前批次中每个源语言句子的实际长度列表
        @returns enc_hiddens (Tensor): 隐藏状态张量，形状为 (b, src_len, h*2)，其中
                                       b = 批次大小，src_len = 源语言句子的最大长度，
                                       h = 隐藏状态维度。
        @returns dec_init_state (tuple(Tensor, Tensor)): 由两个张量组成的元组，
                                                         分别表示解码器的初始隐藏状态和初始细胞状态。
        """
        enc_hiddens, dec_init_state = None, None

        ### 在此处编写代码（约 11 行）
        ### TODO：
        ###     1. 使用源语言的模型词嵌入层，构建源语言句子的张量 `X`，
        ###        其形状为 (src_len, b, e)。
        ###        src_len = 源语言句子的最大长度，b = 批次大小，
        ###        e = 词嵌入维度。
        ###        注意，编码器没有初始隐藏状态或初始细胞状态。
        ###
        ###     2. 应用 post_embed_cnn 层。
        ###        在将 X 输入 CNN 之前，首先使用 torch.permute 将
        ###        X 的形状调整为 (b, e, src_len)。
        ###        得到 CNN 的输出后，记得再次使用 torch.permute，
        ###        将 X 恢复为原来的形状。
        ###
        ###     3. 将 `X` 输入编码器，计算 `enc_hiddens`、`last_hidden` 和 `last_cell`。
        ###        - 在应用编码器之前，需要先对 X 使用 `pack_padded_sequence` 函数。
        ###        - 应用编码器之后，需要对 enc_hiddens 使用 `pad_packed_sequence` 函数。
        ###        - 注意，编码器返回的张量形状为 (src_len, b, h*2)，
        ###          而我们希望返回的 `enc_hiddens` 形状为 (b, src_len, h*2)。
        ###
        ###     4. 计算 `dec_init_state` = (init_decoder_hidden, init_decoder_cell)：
        ###        - `init_decoder_hidden`：
        ###            `last_hidden` 是一个形状为 (2, b, h) 的张量。
        ###            第一个维度对应正向和反向两个方向。
        ###            将正向和反向的张量拼接，得到形状为 (b, 2*h) 的张量。
        ###            对该张量应用 h_projection 层，计算 init_decoder_hidden。
        ###            这对应 PDF 中的 h_0^{dec}。
        ###            其中 b = 批次大小，h = 隐藏状态维度。
        ###
        ###        - `init_decoder_cell`：
        ###            `last_cell` 是一个形状为 (2, b, h) 的张量。
        ###            第一个维度对应正向和反向两个方向。
        ###            将正向和反向的张量拼接，得到形状为 (b, 2*h) 的张量。
        ###            对该张量应用 c_projection 层，计算 init_decoder_cell。
        ###            这对应 PDF 中的 c_0^{dec}。
        ###            其中 b = 批次大小，h = 隐藏状态维度。
        ###
        ### 请参考以下文档，实现过程中可能需要使用其中的一些函数：
        ###     在将填充后的序列 X 输入编码器之前，对其进行打包：
        ###         https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
        ###     对编码器返回的打包序列 enc_hiddens 进行填充还原：
        ###         https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html
        ###     张量拼接：
        ###         https://pytorch.org/docs/stable/generated/torch.cat.html
        ###     张量维度重排：
        ###         https://pytorch.org/docs/stable/generated/torch.permute.html

        # 1. 使用源语言的模型词嵌入层，构建源语言句子的张量 `X`
        # 参考nn.Embedding的调用
        # embedding = nn.Embedding(10, 3)
        # input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        # embedding(input)
        X = self.model_embeddings.source(source_padded)

        # 2. 应用 post_embed_cnn 层
        X = torch.permute(X, (1, 2, 0))
        X = self.post_embed_cnn(X)
        X = torch.permute(X, (2, 0, 1))

        #  3. 将 `X` 输入编码器，计算 `enc_hiddens`、`last_hidden` 和 `last_cell`
        X = pack_padded_sequence(X, source_lengths)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(X)
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens)
        # 调整为 (b, src_len, h*2)
        enc_hiddens = enc_hiddens.permute(1, 0, 2)

        # 4. 计算 `dec_init_state` = (init_decoder_hidden, init_decoder_cell)：
        # 拼接
        last_hidden = torch.cat([last_hidden[0], last_hidden[1]], dim=1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], dim=1)
        # 投影
        init_decoder_hidden = self.h_projection(last_hidden)  # h_0^{dec}
        init_decoder_cell = self.c_projection(last_cell)  # c_0^{dec}

        dec_init_state = (init_decoder_hidden, init_decoder_cell)



        ### 代码结束


        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
               dec_init_state: Tuple[torch.Tensor, torch.Tensor],
               target_padded: torch.Tensor) -> torch.Tensor:
        """计算一个批次的组合输出向量。

        @param enc_hiddens (Tensor): 编码器隐藏状态，形状为 (b, src_len, h*2)，其中
                                     b = 批次大小，src_len = 源语言句子的最大长度，
                                     h = 隐藏状态维度。
        @param enc_masks (Tensor): 句子掩码张量，形状为 (b, src_len)，其中
                                   b = 批次大小，src_len = 源语言句子的最大长度。
        @param dec_init_state (tuple(Tensor, Tensor)): 解码器的初始隐藏状态和细胞状态。
        @param target_padded (Tensor): 填充后的标准答案目标语言句子，
                                      形状为 (tgt_len, b)，其中
                                      tgt_len = 目标语言句子的最大长度，
                                      b = 批次大小。

        @returns combined_outputs (Tensor): 组合输出张量，
                                             形状为 (tgt_len, b, h)，其中
                                             tgt_len = 目标语言句子的最大长度，
                                             b = 批次大小，h = 隐藏状态维度。
        """
        # 去掉目标句子最后一个时间步的 token（最长句子的 <END>）
        target_padded = target_padded[:-1]

        # 初始化解码器状态（隐藏状态和细胞状态）
        dec_state = dec_init_state

        # 将上一时间步的组合输出向量 o_{t-1} 初始化为零
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # 初始化列表，用于收集每个时间步的组合输出向量 o_t
        combined_outputs = []

        ### YOUR CODE HERE (~9 Lines)
        ### TODO:
        ###     1. 对 enc_hiddens 应用注意力投影层，得到 enc_hiddens_proj，
        ###        其形状应为 (b, src_len, h)，其中
        ###        b = 批次大小，src_len = 源语言句子的最大长度，
        ###        h = 隐藏状态维度。
        ###        这对应于 PDF 中将 W_{attProj} 应用于 h^enc。
        ###
        ###     2. 使用目标语言的模型词嵌入层，构建目标语言句子的张量 Y，
        ###        形状为 (tgt_len, b, e)，其中
        ###        tgt_len = 目标语言句子的最大长度，
        ###        b = 批次大小，e = 词嵌入维度。
        ###
        ###     3. 使用 torch.split 函数，沿 Y 的时间维度进行迭代。
        ###        循环中得到的 Y_t 形状为 (1, b, e)。
        ###        - 将 Y_t 压缩为形状 (b, e) 的张量。
        ###        - 沿最后一个维度拼接 Y_t 和 o_prev，构建 Ybar_t。
        ###        - 使用 step 函数计算解码器下一时间步的状态
        ###          （隐藏状态和细胞状态），以及新的组合输出 o_t。
        ###        - 将 o_t 添加到 combined_outputs 列表。
        ###        - 将 o_prev 更新为新的 o_t。
        ###
        ###     4. 使用 torch.stack，将 combined_outputs 从一个包含
        ###        tgt_len 个 (b, h) 张量的列表转换为一个
        ###        形状为 (tgt_len, b, h) 的张量。
        ###
        ### 注意：
        ###     使用 squeeze() 时，务必指定要压缩的维度。
        ###     否则，当 batch_size = 1 时，可能意外删除批次维度。
        ###
        ### 可能用到的函数：
        ###     torch.zeros:
        ###         https://pytorch.org/docs/stable/generated/torch.zeros.html
        ###     torch.split:
        ###         https://pytorch.org/docs/stable/generated/torch.split.html
        ###     torch.squeeze:
        ###         https://pytorch.org/docs/stable/generated/torch.squeeze.html
        ###     torch.cat:
        ###         https://pytorch.org/docs/stable/generated/torch.cat.html
        ###     torch.stack:
        ###         https://pytorch.org/docs/stable/generated/torch.stack.html

        # 1. 对 enc_hiddens 应用注意力投影层，得到 enc_hiddens_proj
        # (b, src_len, 2h) -> (b, src_len, h)
        enc_hiddens_proj = self.att_projection(enc_hiddens)

        # 2. 使用目标语言的模型词嵌入层，构建目标语言句子的张量 Y
        # (tgt_len, b) -> (tgt_len, b, e)
        Y = self.model_embeddings.target(target_padded)

        # 3. 使用 torch.split 函数，沿 Y 的时间维度进行迭代
        for Y_t in torch.split(Y, 1, 0):
            # 将 Y_t 压缩为形状 (b, e) 的张量
            Y_t = Y_t.squeeze(0) # (1, b, e) -> (b, e)
            # 沿最后一个维度拼接 Y_t 和 o_prev，构建 Ybar_t
            Ybar_t = torch.cat((Y_t, o_prev), dim=1)
            # 使用 step 函数计算解码器下一时间步的状态
            dec_state, o_t, _ = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)

            # 将 o_t 添加到 combined_outputs 列表
            combined_outputs.append(o_t)

            # 将 o_prev 更新为新的 o_t
            o_prev = o_t

        # 4. 转换 combined_outputs，# 4. List[Tensor(b, h)]->Tensor(tgt_len, b, h)
        combined_outputs = torch.stack(combined_outputs, dim=0)



        ### 代码结束

        return combined_outputs

    def step(self, Ybar_t: torch.Tensor,
             dec_state: Tuple[torch.Tensor, torch.Tensor],
             enc_hiddens: torch.Tensor,
             enc_hiddens_proj: torch.Tensor,
             enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """执行 LSTM 解码器的一个前向计算步骤，包括注意力计算。

        @param Ybar_t (Tensor): [Y_t, o_prev] 拼接后的张量，形状为 (b, e + h)。
                                它是解码器的输入，其中 b = 批次大小，
                                e = 词嵌入维度，h = 隐藏状态维度。
        @param dec_state (tuple(Tensor, Tensor)): 包含两个张量的元组，
                                两个张量的形状均为 (b, h)，其中
                                b = 批次大小，h = 隐藏状态维度。
                                第一个张量是解码器上一时间步的隐藏状态，
                                第二个张量是解码器上一时间步的细胞状态。
        @param enc_hiddens (Tensor): 编码器隐藏状态张量，形状为 (b, src_len, h * 2)，
                                     其中 b = 批次大小，src_len = 源语言句子的最大长度，
                                     h = 隐藏状态维度。
        @param enc_hiddens_proj (Tensor): 编码器隐藏状态经过投影后的张量，
                                          特征维度从 (h * 2) 投影为 h。
                                          张量形状为 (b, src_len, h)，其中
                                          b = 批次大小，src_len = 源语言句子的最大长度，
                                          h = 隐藏状态维度。
        @param enc_masks (Tensor): 句子掩码张量，形状为 (b, src_len)，
                                   其中 b = 批次大小，
                                   src_len = 源语言句子的最大长度。

        @returns dec_state (tuple(Tensor, Tensor)): 包含两个张量的元组，
                                两个张量的形状均为 (b, h)，其中
                                b = 批次大小，h = 隐藏状态维度。
                                第一个张量是解码器当前时间步的新隐藏状态，
                                第二个张量是解码器当前时间步的新细胞状态。
        @returns combined_output (Tensor): 当前时间步 t 的组合输出张量，
                                           形状为 (b, h)，其中
                                           b = 批次大小，h = 隐藏状态维度。
        @returns e_t (Tensor): 形状为 (b, src_len) 的张量，表示注意力分数。
                               注意：你不会在该函数外部使用这个张量。
                               返回它只是为了对你的实现进行合理性检查。
        """
        combined_output = None

        ### YOUR CODE HERE (~3 Lines)
        ### TODO:
        ###     1. 将 Ybar_t 和 dec_state 输入解码器，得到新的 dec_state。
        ###     2. 将 dec_state 拆分为两个部分 (dec_hidden, dec_cell)。
        ###     3. 计算注意力分数 e_t，其张量形状为 (b, src_len)。
        ###        注意：b = 批次大小，src_len = 源语言句子的最大长度，
        ###        h = 隐藏状态维度。
        ###
        ###     提示：
        ###         - dec_hidden 的形状为 (b, h)，对应 PDF 中的
        ###           h_t^dec（批量形式）。
        ###         - enc_hiddens_proj 的形状为 (b, src_len, h)，对应
        ###           PDF 中的 W_{attProj} h^enc（批量形式）。
        ###         - 使用批量矩阵乘法 torch.bmm 计算 e_t
        ###           （注意输入和输出的形状！）。
        ###         - 为了使张量形状满足 bmm 的要求，
        ###           需要进行一些 squeeze 和 unsqueeze 操作。
        ###         - 使用 squeeze() 时，务必指定需要压缩的维度。
        ###           否则，当 batch_size = 1 时，可能意外删除批次维度。
        ###
        ### 使用以下文档实现上述功能：
        ###     批量矩阵乘法：
        ###         https://pytorch.org/docs/stable/generated/torch.bmm.html
        ###     张量增加维度：
        ###         https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
        ###     张量压缩维度：
        ###         https://pytorch.org/docs/stable/generated/torch.squeeze.html

        # 1. 将 Ybar_t 和 dec_state 输入解码器，得到新的 dec_state
        dec_state = self.decoder(Ybar_t, dec_state)

        # 2. 将 dec_state 拆分为两个部分 (dec_hidden, dec_cell)
        dec_hidden, dec_cell = dec_state

        # 3. 计算注意力分数 e_t，形状为 (b, src_len)
        e_t = torch.bmm(enc_hiddens_proj, dec_hidden.unsqueeze(2)).squeeze(2)


        ### END YOUR CODE

        # Question 1(g): Written Answer
        #
        # 掩码将 <pad> 位置对应的注意力分数设置为负无穷。
        # 经过 Softmax 后，这些位置的注意力权重变为 0，
        # 而有效 token 的注意力权重被归一化，总和为 1。
        # 因此，填充位置不会对最终的注意力上下文向量产生贡献。
        #
        # 使用掩码是必要的，因为 <pad> 仅用于统一批次中句子的长度，
        # 不包含实际的语义信息。
        # 如果不使用掩码，模型可能将注意力分配给这些无效位置，
        # 从而干扰注意力计算。

        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        ### YOUR CODE HERE (~6 Lines)
        ### TODO:
        ###     1. 对 e_t 应用 softmax，得到 alpha_t。
        ###     2. 使用批量矩阵乘法，将 alpha_t 与 enc_hiddens 相乘，
        ###        得到注意力输出向量 a_t。
        ###           - alpha_t 的形状为 (b, src_len)。
        ###           - enc_hiddens 的形状为 (b, src_len, 2h)。
        ###           - a_t 的形状应为 (b, 2h)。
        ###           - 需要使用 squeeze 和 unsqueeze 操作。
        ###        注意：b = 批次大小，src_len = 源语言句子的最大长度，
        ###        h = 隐藏状态维度。
        ###
        ###     3. 将 dec_hidden 与 a_t 拼接，计算张量 U_t。
        ###     4. 对 U_t 应用组合输出投影层，计算张量 V_t。
        ###     5. 先应用 Tanh 函数，再应用 Dropout 层，计算张量 O_t。
        ###
        ### 使用以下文档实现上述功能：
        ###     Softmax：
        ###         https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
        ###     批量矩阵乘法：
        ###         https://pytorch.org/docs/stable/generated/torch.bmm.html
        ###     张量视图：
        ###         https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
        ###     张量拼接：
        ###         https://pytorch.org/docs/stable/generated/torch.cat.html
        ###     Tanh：
        ###         https://pytorch.org/docs/stable/generated/torch.tanh.html

        # 1. 对 e_t 应用 softmax，得到 alpha_t
        alpha_t = F.softmax(e_t, dim=1)

        # 2. 使用批量矩阵乘法，将 alpha_t 与 enc_hiddens 相乘, 得到注意力输出向量 a_t
        a_t = torch.bmm(alpha_t.unsqueeze(1), enc_hiddens).squeeze(1)  # (b, 2h)

        # 3. 将 dec_hidden 与 a_t 拼接，计算张量 U_t
        # dec_hidden 形状 (b, h)
        U_t = torch.cat((dec_hidden, a_t), dim=1) # (b, 3h)

        # 4. 对 U_t 应用组合输出投影层，计算张量 V_t
        V_t = self.combined_output_projection(U_t) # (b, h)

        # 5. 先应用 Tanh 函数，再应用 Dropout 层，计算张量 O_t
        O_t = self.dropout(torch.tanh(V_t)) # (b, h)



        ### END YOUR CODE

        combined_output = O_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ 为编码器的隐藏状态生成句子掩码。

        @param enc_hiddens (Tensor): 编码器输出张量，形状为 (b, src_len, 2*h)，
                                     其中 b = 批次大小，src_len = 源语言句子的最大长度，
                                     h = 隐藏状态维度。
        @param source_lengths (List[int]): 批次中每个源语言句子的实际长度。

        @returns enc_masks (Tensor): 句子掩码张量，形状为 (b, src_len)，
                                     其中 src_len = 源语言句子的最大长度，
                                     h = 隐藏状态维度。
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[
        Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.target(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _ = self.step(x, h_tm1,
                                                exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = torch.div(top_cand_hyp_pos, len(self.vocab.tgt), rounding_mode='floor')
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
