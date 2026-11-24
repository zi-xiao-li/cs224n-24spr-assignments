
import argparse
import random

import torch
from torch.utils.data import Dataset

# [part e]

# CharCorruptionDataset 是一个数据集类，用于生成简化版
# Span Corruption（文本片段破坏）预训练任务的样本。
# 请根据以下说明实现 __getitem__ 函数。
# 不要修改 __init__ 或 __getitem__ 函数的签名。

# 为获得满分，请确保完整实现以下规范。
# 下面列出了完整实现必须满足的所有要求。

# --------------
# 掩码规范（Masking Specification）

# __getitem__ 函数接收一个索引，并返回一个数据样本 (x, y)。
# x 和 y 均为长度为 self.block_size 的 Long Tensor。
# x 编码输入序列，y 编码输出序列。

# 0. 使用 __getitem__ 的 idx 参数，从 self.data 中
#    获取指定索引处的元素。
#    我们将得到的数据条目称为一个文档（document）。

# 1. 随机截断文档，使其长度不少于 4 个字符，
#    且不超过 int(self.block_size*7/8) 个字符。

# - 重要：你可以自行决定如何实现随机截断，但为了获得满分，
#   必须确保截断长度是随机选择的。
#   从 4 到 int(self.block_size*7/8) 的每个可能长度
#   都必须有被选中的机会。

# 2. 现在，将截断后的文档划分为三个子字符串：
#
#     [prefix] [masked_content] [suffix]
#
#   换句话说，选择三个字符串 prefix、masked_content 和 suffix，
#   使它们满足：
#
#     prefix + masked_content + suffix = [截断后的文档]
#
#   [masked_content] 的长度应该是随机的，
#   并且平均约为截断后文档长度的 1/4。

# - 重要：你可以自行决定如何实现这一操作，但为了获得满分，
#   必须确保 [masked_content] 的长度是随机选择的，
#   即它既有机会大于，也有机会小于截断后文档长度的 1/4。

# 3. 将这些子字符串重新排列为以下形式：
#
#     [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]
#
#   生成的字符串称为 masked_string，用作输出样本。
#   其中，MASK_CHAR 是掩码字符；
#   [pads] 是由重复的 PAD_CHAR 字符组成的字符串，
#   其数量应保证整个字符串长度为 self.block_size + 1。
#
#   直观地说，[masked_content] 会从文档中移除，
#   并在原位置替换为 MASK_CHAR。
#   在 suffix 之后再次出现 MASK_CHAR，
#   随后依次添加被移除的内容和填充字符。

# 4. 现在使用 masked_string 构造输入与输出样本对。
#    具体来说：
#
#    输入字符串：masked_string[:-1]
#    输出字符串：masked_string[1:]
#
#    换句话说，对于每个字符，模型的目标是预测
#    masked_string 中的下一个字符。

# 5. 使用已经定义好的词表，将生成的输入和输出字符串
#    编码为 Long Tensor，并返回得到的数据样本。

# ----------------
# 以下是一些输入与输出样本对 (x, y) 的示例：

#   x: Khatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   y: hatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   x: Jaco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   y: aco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   x: John Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
#   y: ohn Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

class CharCorruptionDataset(Dataset):
    def __init__(self, data, block_size):
        self.MASK_CHAR = "\u2047" # 双问号字符，用作掩码
        self.PAD_CHAR = "\u25A1" # 空心方框字符，用作填充

        chars = list(sorted(list(set(data))))
        assert self.MASK_CHAR not in chars
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        # 两行代码构建了一个简化的字符级 Tokenizer，实现字符与 Token ID 之间的双向转换
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}

        data_size, vocab_size = len(data), len(chars)
        print(f'data has {data_size} characters, {vocab_size} unique.')

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data.split('\n')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO [part e]：参见上方的实现规范
        ### YOUR CODE HERE ###

        # 0. 根据指定索引获取文档
        document = self.data[idx]

        # 1. 随机截断文档
        max_length = min(len(document), int(self.block_size*7/8))
        # 随机选择截断长度
        trunc_length = random.randint(4, max_length)
        # 保留文档前trunc_length个字符
        document = document[:trunc_length]

        # 2. 将截断后的文档划分为三个子字符串：[prefix] [masked_content] [suffix]
        # 它们满足 prefix + masked_content + suffix = [截断后的文档]
        # 掩码长度的期望值约为文档长度的1/4
        mask_length = random.randint(1, max(1, len(document) // 2))

        # 随机选择掩码片段的起始位置
        mask_start = random.randint(0, len(document) - mask_length)

        # 将文档划为三部分
        prefix = document[: mask_start]
        masked_content = document[mask_start: mask_start + mask_length]
        suffix = document[mask_start + mask_length:]

        # 3. 重组文本：[prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]
        masked_string = (
            prefix
            + self.MASK_CHAR
            + suffix
            + self.MASK_CHAR
            + masked_content
        )
        masked_string += self.PAD_CHAR * (self.block_size + 1 - len(masked_string))

        # 4. 使用 masked_string 构造输入与输出样本对
        x = masked_string[:-1]
        y = masked_string[1:]

        # 5. 将字符编码为 Token ID, 并转换为 Long Tensor
        x = torch.tensor(
            [self.stoi[c] for c in x],
            dtype=torch.long
        )
        y = torch.tensor(
            [self.stoi[c] for c in y],
            dtype=torch.long
        )

        return x, y

        ### END YOUR CODE ###



# NameDataset 是用于人物出生地预测任务的数据集类
# 它继承 PyTorch 的 Dataset，负责将原始姓名-出生地数据转换为 GPT 训练所需的 (x, y)
# 可以将原始数据转换为：
# 1）x：包含问题和答案的字符序列
# 2）y：用于计算 Loss 的目标序列，其中问题部分用 PAD_CHAR 填充


# NameDataset 的输入与输出样本对 (x, y) 具有以下形式：

# x: Where was Khatchig Mouradian born?⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
# y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
# x: Where was Jacob Henry Studer born?⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
# y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

# 在 y 中，使用 PAD_CHAR 填充 ⁇[place] 之前的位置，
# 可以避免 Trainer 在训练时优化模型对问题部分（"Where was..."）的预测。

# 注意：NameDataset 应接收 run.py 中定义的 pretraining_dataset 作为输入。
# 这样可以确保 NameDataset 的词表定义与预训练数据集保持一致。

# NameDataset 不需要你实现任何代码。

class NameDataset(Dataset):
    def __init__(self, pretraining_dataset, data):
        self.MASK_CHAR = "\u2047" # 双问号字符，用作掩码
        self.PAD_CHAR = "\u25A1" # 空心方框字符，用作填充
        self.itos = pretraining_dataset.itos
        self.stoi = pretraining_dataset.stoi
        self.block_size = pretraining_dataset.block_size
        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        inp, oup = self.data[idx].split('\t')
        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR
        x = x + self.PAD_CHAR*(self.block_size - len(x))
        y = self.PAD_CHAR*(len(inp)-1) + x[len(inp):]

        x = x[:-1]
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        return x, y


# Code below is strictly for your debugging purposes; feel free to modify
# as desired.

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('dataset_type', help="Type of dataset to sample from."
            "Options: namedata, charcorruption.",
            choices=["namedata", "charcorruption"])
    args = argp.parse_args()

    if args.dataset_type == 'namedata':
        # Even if it hasn't been implemented, we use it to define the vocab
        corruption_dataset = CharCorruptionDataset(
            open('wiki.txt', encoding='utf-8').read(), 128)
        # Make the name dataset
        name_dataset = NameDataset(corruption_dataset,
            open('birth_places_train.tsv', encoding='utf-8').read())
        for _, example in zip(range(4), name_dataset):
            x, y = example
            print('x:', ''.join([name_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([name_dataset.itos[int(c)] for c in y]))
    elif args.dataset_type == 'charcorruption':
        corruption_dataset = CharCorruptionDataset(
            open('wiki.txt', encoding='utf-8').read(), 128)
        for _, example in zip(range(4), corruption_dataset):
            x, y = example
            print('x:', ''.join([corruption_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([corruption_dataset.itos[int(c)] for c in y]))
    else:
        raise ValueError(f"Unknown dataset type in command line args: {args.dataset_type}")
