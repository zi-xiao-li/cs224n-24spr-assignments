
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2022-23：作业 4
utils.py：工具函数
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
Siyan Li <siyanli@stanford.edu>
Moussa KB Doumbouya <moussa@stanford.edu>
"""

from typing import List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import sentencepiece as spm
# nltk.download('punkt')


def pad_sents(sents, pad_token):
    """ 根据批次中最长的句子，对句子列表进行填充。
        填充应添加在每个句子的末尾。

    @param sents (list[list[str]]): 句子列表，其中每个句子
                                    由一个单词列表表示
    @param pad_token (str): 填充标记
    @returns sents_padded (list[list[str]]): 填充后的句子列表。
        比最长句子短的句子需要使用 pad_token 进行填充，
        使批次中的所有句子具有相同的长度。
    """
    sents_padded = []

    ### 在此处编写代码（约 6 行）

    # 找出当前 batch 中最长句子的长度
    max_len = max(len(sent) for sent in sents)

    for sent in sents:
        # 计算当前句子需要填充的 token 数量
        num_padding = max_len - len(sent)
        # 在句子末尾填充 pad_token
        padded_sent = sent + [pad_token] * num_padding
        # 将填充后的句子加入结果列表
        sents_padded.append(padded_sent)

    ### 代码结束

    return sents_padded


def read_corpus(file_path, source, vocab_size=2500):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    @param vocab_size (int): number of unique subwords in
        vocabulary when reading and tokenizing
    """
    data = []
    sp = spm.SentencePieceProcessor()
    sp.load('{}.model'.format(source))

    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            subword_tokens = sp.encode_as_pieces(line)
            # only append <s> and </s> to the target sentence
            if source == 'tgt':
                subword_tokens = ['<s>'] + subword_tokens + ['</s>']
            data.append(subword_tokens)

    return data


def autograder_read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = nltk.word_tokenize(line)
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


