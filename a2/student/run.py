
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2023-2024：作业 2
run.py：运行依存句法分析器。
Sahil Chopra <schopra8@stanford.edu>
Haoshen Hong <haoshen@stanford.edu>
"""
from datetime import datetime
import os
import pickle
import math
import time
import argparse

from torch import nn, optim
import torch
from tqdm import tqdm

from parser_model import ParserModel
from utils.parser_utils import minibatches, load_and_preprocess_data, AverageMeter

parser = argparse.ArgumentParser(description='使用 PyTorch 训练神经网络依存句法分析器')
parser.add_argument('-d', '--debug', action='store_true', help='是否进入调试模式')
args = parser.parse_args()

# -----------------
# 主要函数
# -----------------
def train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005):
    """ 训练神经网络依存句法分析器。

    @param parser (Parser): 神经网络依存句法分析器
    @param train_data ():
    @param dev_data ():
    @param output_path (str): 保存模型权重和结果的路径
    @param batch_size (int): 每个批次的样本数量
    @param n_epochs (int): 训练轮数
    @param lr (float): 学习率
    """
    best_dev_UAS = 0

    ### YOUR CODE HERE (~2-7 lines)
    ### TODO:
    ###     1) 创建 Adam 优化器，并赋值给变量 `optimizer`
    ###     2) 创建交叉熵损失函数，并赋值给变量 `loss_func`
    ###        使用 `mean` 归约方式（默认值）
    ###
    ### 提示：使用 `parser.model.parameters()` 将需要优化的模型参数传递给优化器。
    ### 以下文档可供参考：
    ###     Adam 优化器：https://pytorch.org/docs/stable/optim.html
    ###     交叉熵损失函数：https://pytorch.org/docs/stable/nn.html#crossentropyloss

    optimizer = optim.Adam(parser.model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss(reduction='mean') # 损失取平均

    ### END YOUR CODE

    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        dev_UAS = train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size)
        if dev_UAS > best_dev_UAS:
            best_dev_UAS = dev_UAS
            print("New best dev UAS! Saving model.")
            torch.save(parser.model.state_dict(), output_path)
        print("")




def train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size):
    """ 对神经网络依存句法分析器进行一个 epoch 的训练。

    注意：在 PyTorch 中，可以通过指定训练模式 `model.train()` 或评估模式
    `model.eval()` 来区分训练和测试，并相应地自动启用或关闭 Dropout 层。

    @param parser (Parser): 神经网络依存句法分析器
    @param train_data ():
    @param dev_data ():
    @param optimizer (nn.Optimizer): Adam 优化器
    @param loss_func (nn.CrossEntropyLoss): 交叉熵损失函数
    @param batch_size (int): 批次大小

    @return dev_UAS (float): 开发集上的无标签依存准确率（UAS）
    """
    parser.model.train()  # 将模型设置为训练模式，即启用 Dropout 层
    n_minibatches = math.ceil(len(train_data) / batch_size)
    loss_meter = AverageMeter()

    with tqdm(total=(n_minibatches)) as prog:
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            optimizer.zero_grad()   # 清除优化器所管理参数上累积的梯度
            loss = 0.  # 在此处存储当前批次的损失
            train_x = torch.from_numpy(train_x).long()
            train_y = torch.from_numpy(train_y.nonzero()[1]).long()

            ### YOUR CODE HERE (~4-10 lines)
            ### TODO:
            ###     1) 将 train_x 输入模型进行前向传播，得到 `logits`
            ###     2) 使用参数 `loss_func` 调用 PyTorch 的 CrossEntropyLoss 损失函数。
            ###        该函数接收 `logits` 和 `train_y` 作为输入，计算
            ###        softmax(`logits`) 与 `train_y` 之间的交叉熵损失。
            ###        请记住，softmax(`logits`) 是模型的预测结果
            ###        （即 PDF 中的 ŷ）。
            ###     3) 对损失进行反向传播
            ###     4) 使用优化器执行一次参数更新
            ### 以下文档可供参考：
            ###     优化器参数更新：https://pytorch.org/docs/stable/optim.html#optimizer-step

            # 1. 前向传播，得到模型预测的 logits
            logits = parser.model(train_x)

            # 2. 计算交叉熵损失
            # 注意：交叉熵损失函数已经包含softmax
            loss = loss_func(logits, train_y)

            # 3. 反向传播，计算梯度
            loss.backward()

            # 4. 使用 Adam 优化器更新模型参数
            optimizer.step()


            ### END YOUR CODE
            prog.update(1)
            loss_meter.update(loss.item())

    print("Average Train Loss: {}".format(loss_meter.avg))

    print("Evaluating on dev set",)
    parser.model.eval()  # 将模型设置为评估模式，即关闭 Dropout 层
    dev_UAS, _ = parser.parse(dev_data)
    print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
    return dev_UAS



if __name__ == "__main__":
    debug = args.debug

    assert (torch.__version__.split(".") >= ["1", "0", "0"]), "Please install torch version >= 1.0.0"

    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)

    start = time.time()
    model = ParserModel(embeddings)
    parser.model = model
    print("took {:.2f} seconds\n".format(time.time() - start))

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005)

    if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        parser.model.load_state_dict(torch.load(output_path))
        print("Final evaluation on test set",)
        parser.model.eval()
        UAS, dependencies = parser.parse(test_data)
        print("- test UAS: {:.2f}".format(UAS * 100.0))
        print("Done!")
