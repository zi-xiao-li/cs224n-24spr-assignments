
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2023-2024：作业 2
parser_model.py：用于依存句法分析的前馈神经网络
Sahil Chopra <schopra8@stanford.edu>
Haoshen Hong <haoshen@stanford.edu>
"""
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParserModel(nn.Module):
    """ 具有嵌入层和两个隐藏层的前馈神经网络。
    ParserModel 将预测对于给定的部分句法分析状态，
    下一步应该执行哪种转换操作。

    PyTorch 注意事项：
        - 注意，"ParserModel" 是 "nn.Module" 类的子类。在 PyTorch 中，
          所有神经网络都是 "nn.Module" 的子类。
        - "__init__" 方法用于定义所有网络层和参数
          （嵌入层、线性层、Dropout 层等）。
        - 创建类的新实例时，"__init__" 会被自动调用。例如，
          当你编写 "m = ParserModel()" 时。
        - ParserModel 的其他方法可以访问带有 "self." 前缀的变量。因此，
          如果希望在 ParserModel 的其他方法中使用某些网络层、变量值等，
          应为它们添加 "self." 前缀。
        - 关于 "nn.Module" 的更多文档，请参阅 https://pytorch.org/docs/stable/nn.html。
    """
    def __init__(self, embeddings, n_features=36,
        hidden_size=200, n_classes=3, dropout_prob=0.5):
        """ 初始化句法分析模型。

        @param embeddings (ndarray): 词嵌入矩阵，形状为 (num_words, embedding_size)
        @param n_features (int): 输入特征的数量
        @param hidden_size (int): 隐藏层神经元的数量
        @param n_classes (int): 输出类别的数量
        @param dropout_prob (float): Dropout 概率
        """
        super(ParserModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.embeddings = nn.Parameter(torch.tensor(embeddings))

        ### YOUR CODE HERE (~9-10 Lines)
        ### TODO:
        ###     1) 将 `self.embed_to_hidden_weight` 和 `self.embed_to_hidden_bias` 声明为 `nn.Parameter`。
        ###        使用 `nn.init.xavier_uniform_` 函数初始化权重，
        ###        使用 `nn.init.uniform_` 函数及其默认参数初始化偏置。
        ###     2) 构建 `self.dropout` 层。
        ###     3) 将 `self.hidden_to_logits_weight` 和 `self.hidden_to_logits_bias` 声明为 `nn.Parameter`。
        ###        使用 `nn.init.xavier_uniform_` 函数初始化权重，
        ###        使用 `nn.init.uniform_` 函数及其默认参数初始化偏置。
        ###
        ### 注意：可训练变量使用 `nn.Parameter` 声明。这是一个常用的 API，
        ###       可以将张量注册为模型参数，使其参与计算图，并支持根据梯度更新参数。
        ###       这里，我们使用 Xavier 均匀初始化方法初始化权重。
        ###       实验表明，与普通随机均匀初始化相比，
        ###       这种方法能够为神经网络训练提供更合适的初始权重。
        ###       更多详细信息，请参阅这篇优秀的博客文章：
        ###             http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
        ###
        ### 以下文档可供参考：
        ###     nn.Parameter: https://pytorch.org/docs/stable/nn.html#parameters
        ###     初始化方法: https://pytorch.org/docs/stable/nn.init.html
        ###     Dropout: https://pytorch.org/docs/stable/nn.html#dropout-layers
        ###
        ### 请参阅 PDF 作业说明中的提示。

        # 第一层：输出词向量-》隐藏层

        # 创建第一层权重矩阵 W，形状为 (n_features * embed_size, hidden_size)
        # nn.Parameter 将 Tensor 注册为模型的可训练参数
        # torch.empty 只创建指定形状的 Tensor，不初始化其数值
        self.embed_to_hidden_weight = nn.Parameter(
            torch.empty(self.n_features * self.embed_size, self.hidden_size)
        )
        # 使用 Xavier 均匀分布初始化权重 W
        nn.init.xavier_uniform_(self.embed_to_hidden_weight)

        # 创建第一层偏置向量 b1，长度等于隐藏层神经元数量
        self.embed_to_hidden_bias = nn.Parameter(torch.empty(self.hidden_size))
        # 使用默认均匀分布 U(0, 1) 初始化偏置 b1
        nn.init.uniform_(self.embed_to_hidden_bias)


        # Dropout层

        self.dropout = nn.Dropout(p=self.dropout_prob)

        # 第二层：隐藏层-》输出层

        # 创建第二层权重矩阵 U，形状为 (hidden_size, n_classes)
        # n_classes=3，分别对应 SHIFT、LEFT-ARC、RIGHT-ARC
        self.hidden_to_logits_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.n_classes)
        )
        nn.init.xavier_uniform_(self.hidden_to_logits_weight)

        # 创建第二层偏置向量 b2，长度为 n_classes（默认3）
        self.hidden_to_logits_bias = nn.Parameter(torch.empty(self.n_classes))
        nn.init.uniform_(self.hidden_to_logits_bias)


        ### END YOUR CODE


    def embedding_lookup(self, w):
        """ 使用 `w` 从嵌入矩阵 `self.embeddings` 中选取嵌入向量。

            @param w (Tensor): 单词索引构成的输入张量，形状为 (batch_size, n_features)

            @return x (Tensor): `w` 中各单词对应的嵌入向量组成的张量，
                                形状为 (batch_size, n_features * embed_size)
        """

        ### YOUR CODE HERE (~1-4 Lines)
        ### TODO:
        ###     1) 对于 `w` 中的每个索引 `i`，从 self.embeddings 中选取第 `i` 个向量
        ###     2) 如有必要，使用 `view` 函数调整张量形状
        ###
        ### 注意：所有嵌入向量都堆叠存储在一个矩阵中。模型接收一个由整数索引组成的列表，
        ###       这些索引表示一个单词序列，然后调用此查找函数，
        ###       将索引序列映射为对应的嵌入向量序列。
        ###
        ###       本题旨在考查你对嵌入查找（Embedding Lookup）的理解，
        ###       因此不要使用 nn.Embedding 等高级 API
        ###       （因为我们正是要求你自己实现这一功能！）。
        ###       请注意张量的形状，并在必要时调整形状。
        ###       运行代码之前，务必清楚每个张量的形状！
        ###
        ### PyTorch 提供了一些有用的 API，本题可以使用其中任意一种
        ### （nn.Embedding 除外）。以下文档可能有所帮助：
        ###     Index select: https://pytorch.org/docs/stable/torch.html#torch.index_select
        ###     Gather: https://pytorch.org/docs/stable/torch.html#torch.gather
        ###     View: https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        ###     Flatten: https://pytorch.org/docs/stable/generated/torch.flatten.html

        # 根据特征索引 w，从嵌入矩阵中查找对应的向量
        # 输出形状：(batch_size, n_features, embed_size)
        x = self.embeddings[w]

        # 将每个样本的 n_features 个嵌入向量拼接成一个长向量
        # 输出形状：(batch_size, n_features * embed_size)
        x = x.reshape(w.shape[0], -1)


        ### END YOUR CODE
        return x

    def forward(self, w):
        """ 执行模型的前向传播。
            注意：这里不应用 softmax 函数，因为损失函数 nn.CrossEntropyLoss 已经包含了 softmax 的计算。

            PyTorch 注意事项：
                - 每个 nn.Module 对象（PyTorch 模型）都有一个 forward 函数。
                - 当你将 nn.Module 应用于输入张量 w 时，就会对该张量执行 forward 函数。
                  例如，假设你创建了一个 ParserModel 实例，并按以下方式将其应用于 w：
                      model = ParserModel()
                      output = model(w)  # 这会调用 forward 函数
                  forward 函数将以 w 为输入执行，其结果存储在 output 变量中。
                - 更多详细信息请参阅：https://pytorch.org/docs/stable/nn.html#torch.nn.Module.forward

            @param w (Tensor): 输入 token 索引张量，形状为 (batch_size, n_features)
            @return logits (Tensor): 神经网络各层计算后的预测输出，未经 softmax 处理，
                                     形状为 (batch_size, n_classes)
        """

        ### YOUR CODE HERE (~3-5 lines)
        ### TODO:
        ###     按照 PDF 作业说明完成前向传播计算。
        ###     此外，在 ReLU 函数之后加入 __init__ 中定义的 dropout 层。
        ###
        ### 注意：这里不对 logits 应用 softmax，
        ###       因为损失函数 torch.nn.CrossEntropyLoss 会更高效地完成相关计算。
        ###
        ### 以下文档可供参考：
        ###     矩阵乘法：https://pytorch.org/docs/stable/torch.html#torch.matmul
        ###     ReLU：https://pytorch.org/docs/stable/nn.html?highlight=relu#torch.nn.functional.relu

        # 根据词索引查找词向量，并拼接为输入向量 x
        x = self.embedding_lookup(w)

        # 第一层：线性变换 + ReLU 激活
        h = F.relu(torch.matmul(x, self.embed_to_hidden_weight)
                   + self.embed_to_hidden_bias)

        # Dropout：训练时随机将部分隐藏层神经元的输出置零
        h = self.dropout(h)

        # 第二层：线性变换，得到三个类别的 logits
        logits = torch.matmul(h, self.hidden_to_logits_weight) \
                  + self.hidden_to_logits_bias

        ### END YOUR CODE
        return logits


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simple sanity check for parser_model.py')
    parser.add_argument('-e', '--embedding', action='store_true', help='sanity check for embeding_lookup function')
    parser.add_argument('-f', '--forward', action='store_true', help='sanity check for forward function')
    args = parser.parse_args()

    embeddings = np.zeros((100, 30), dtype=np.float32)
    model = ParserModel(embeddings)

    def check_embedding():
        inds = torch.randint(0, 100, (4, 36), dtype=torch.long)
        selected = model.embedding_lookup(inds)
        assert np.all(selected.data.numpy() == 0), "The result of embedding lookup: " \
                                      + repr(selected) + " contains non-zero elements."

    def check_forward():
        inputs =torch.randint(0, 100, (4, 36), dtype=torch.long)
        out = model(inputs)
        expected_out_shape = (4, 3)
        assert out.shape == expected_out_shape, "The result shape of forward is: " + repr(out.shape) + \
                                                " which doesn't match expected " + repr(expected_out_shape)

    if args.embedding:
        check_embedding()
        print("Embedding_lookup sanity check passes!")

    if args.forward:
        check_forward()
        print("Forward sanity check passes!")
