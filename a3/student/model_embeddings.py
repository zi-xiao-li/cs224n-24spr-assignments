#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2024 春季：作业 3
model_embeddings.py：神经机器翻译（NMT）模型的词嵌入
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Vera Lin <veralin@stanford.edu>
Siyan Li <siyanli@stanford.edu>
"""

import torch.nn as nn


class ModelEmbeddings(nn.Module):
    """
    将输入单词转换为对应词嵌入的类。
    """

    def __init__(self, embed_size, vocab):
        """
        初始化词嵌入层。

        @param embed_size (int): 词嵌入大小（维度）
        @param vocab (Vocab): 包含源语言和目标语言词表的词表对象
                              具体说明参见 vocab.py。
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size

        # 默认值
        self.source = None
        self.target = None

        src_pad_token_idx = vocab.src['<pad>']
        tgt_pad_token_idx = vocab.tgt['<pad>']

        ### 在此处编写代码（约 2 行）
        ### TODO - 初始化以下变量：
        ###     self.source（源语言的 Embedding 层）
        ###     self.target（目标语言的 Embedding 层）
        ###
        ### 注意：
        ###     1. `vocab` 对象包含两个词表：
        ###            `vocab.src`：源语言词表
        ###            `vocab.tgt`：目标语言词表
        ###     2. 可以通过以下方式获取指定词表的大小：
        ###             `len(vocab.<specific_vocabulary>)`
        ###     3. 创建 Embedding 层时，记得指定对应词表中
        ###        填充标记（padding token）的索引。
        ###
        ### 请参考以下文档，正确初始化这些变量：
        ###     Embedding 层：
        ###         https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

        self.source = nn.Embedding(len(vocab.src), embed_size, padding_idx=src_pad_token_idx)
        self.target = nn.Embedding(len(vocab.tgt), embed_size, padding_idx=tgt_pad_token_idx)


        ### 代码结束
