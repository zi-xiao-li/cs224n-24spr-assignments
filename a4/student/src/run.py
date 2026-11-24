
import random
import argparse

import dataset
import models
import trainer
import utils

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

random.seed(0)

argp = argparse.ArgumentParser()
argp.add_argument('function', help="选择 pretrain、finetune 或 evaluate")
argp.add_argument('variant', help="选择 vanilla 或 rope")
argp.add_argument('pretrain_corpus_path', default=None)
argp.add_argument('--reading_params_path', default=None)
argp.add_argument('--writing_params_path', default=None)
argp.add_argument('--finetune_corpus_path', default=None)
argp.add_argument('--eval_corpus_path', default=None)
argp.add_argument('--outputs_path', default=None)
argp.add_argument('--pretrain_lr', default=6e-3, type=float)
argp.add_argument('--finetune_lr', default=6e-4, type=float)
argp.add_argument('--tb_expt_name', help='用于 TensorBoard 日志的调试字符串。',
                  default='run')
args = argp.parse_args()

# 自动选择计算设备
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available() and args.variant == 'vanilla':
    device = torch.device('mps')

# TensorBoard 训练日志
writer = SummaryWriter(log_dir='expt/%s/%s_%s_pt_lr_%f_ft_lr_%f' % (
    args.function,
    args.tb_expt_name,
    args.variant,
    args.pretrain_lr,
    args.finetune_lr))

# 保持 block_size 为 128
# 为什么即使不进行预训练，也始终需要提供预训练语料？
# 这是因为我们通过这种变通方法，确保始终使用相同的词表。
# 也就是说，字符到整数索引的映射保持一致，
# 并且词表是根据预训练语料构建的。

# * 意思就是：就是让 NameDataset 直接复用已经建立的字符映射，而不是重新构建一个新词表。
# * 这也保证了 GPT 输入层、输出层以及评估阶段使用一致的 Token ID 定义。
block_size = 128
text = open(args.pretrain_corpus_path, encoding='utf-8').read() # 读取 Wikipedia 预训练语料
pretrain_dataset = dataset.CharCorruptionDataset(text, block_size) # 创建 CharCorruptionDataset 实例

# 不建议修改以下超参数，因为它们已经被验证能够正常工作。
# Vanilla 模型和 RoPE 模型均使用这些超参数。
mconf = models.GPTConfig(
    pretrain_dataset.vocab_size,
    pretrain_dataset.block_size,
    n_layer=4,
    n_head=8,
    n_embd=256)

# 定义模型。
# 注意：模型应当移动到第 30–34 行定义的设备上。

# 3. 初始化 GPT 模型
# 根据模型配置 mconf 创建 GPT 模型，并将模型移动到指定的计算设备（CPU 或 GPU）上
model = None
if args.variant == 'vanilla':
    # TODO [part c]：在这里创建模型
    ### YOUR CODE HERE ###

    model = models.GPT(mconf).to(device) # 调用 GPT 类创建模型实例

    ### END YOUR CODE ###
elif args.variant == 'rope':
    # TODO [part g]：在这里创建另一种模型
    # 设置 mconf.rope 参数
    ### YOUR CODE HERE ###

    mconf.rope = True # 启用旋转位置编码
    model = models.GPT(mconf).to(device) # 创建 RoPE GPT 模型

    ### END YOUR CODE ###
else:
    raise ValueError("Unknown model variant")

print('Model on device: ', next(model.parameters()).device)

# 执行预训练、微调或评估
if args.function == 'pretrain':
    assert args.writing_params_path is not None
    # TODO [part f]：
    # - 已知条件：
    #     1. 由 args.pretrain_corpus_path 指定的预训练语料
    #     2. 用于保存模型参数的输出路径 args.writing_params_path
    # - 任务目标：
    #     1. 在该语料上预训练模型
    #     2. 将训练后的模型参数保存到 args.writing_params_path

    # - 确保预训练使用以下超参数：
    # 预训练超参数：
    # max_epochs=650
    # batch_size=128
    # learning_rate=args.pretrain_lr
    # lr_decay=True
    # warmup_tokens=512*20
    # final_tokens=650*len(pretrain_dataset)*block_size
    # num_workers=4
    # writer=writer

    ### YOUR CODE HERE ###

    # 1. 配置预训练超参数
    tconf = trainer.TrainerConfig(
        max_epochs=650,
        batch_size=128,
        learning_rate=args.pretrain_lr,
        lr_decay=True,
        warmup_tokens=512 * 20,
        final_tokens=650 * len(pretrain_dataset) * block_size,
        num_workers=4,
        writer=writer
    )

    # 2. 创建 Trainer，使用 wiki.txt 构建的预训练数据集；开始训练
    trainer_instance = trainer.Trainer(model, pretrain_dataset, None, tconf)
    trainer_instance.train()

    # 3. 保存预训练模型参数
    torch.save(model.state_dict(), args.writing_params_path)


    ### END YOUR CODE ###
elif args.function == 'finetune':
    assert args.writing_params_path is not None
    assert args.finetune_corpus_path is not None
    # TODO [part c] [part f]：
    # - 已知条件：
    #     1. 由 args.finetune_corpus_path 指定的微调语料
    #     2. args.reading_params_path：包含预训练模型参数的文件路径；
    #        如果不使用预训练模型进行微调，则为 None
    #     3. 用于保存模型参数的输出路径 args.writing_params_path
    # - 任务目标：
    #     1. 如果指定了 args.reading_params_path，
    #        则将该路径中的模型参数加载到模型中
    #     2. 在该语料上微调模型
    #     3. 将微调后的模型参数保存到 args.writing_params_path
    # - 确保使用以下超参数：
    #     [part d] 不使用预训练模型时的微调超参数：
    #         max_epochs=75
    #         batch_size=256
    #         learning_rate=args.finetune_lr
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #         writer=writer
    #     [part f] 使用预训练模型时的微调超参数：
    #         max_epochs=10
    #         batch_size=256
    #         learning_rate=args.finetune_lr
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #         writer=writer
    #     可以使用 args.reading_params_path 参数，
    #     在两种情况下切换对应的训练轮数。

    ### YOUR CODE HERE ###

    # 1. 读取人物姓名与出生地训练数据
    text = open(args.finetune_corpus_path, encoding='utf-8').read()

    # 2. 构建训练数据集
    train_dataset = dataset.NameDataset(pretrain_dataset, text)

    # 3. 判断是否加载预训练权重
    if args.reading_params_path is not None:
        # Part 3(f)：加载预训练参数，然后微调 10 Epochs
        model.load_state_dict(
            torch.load(args.reading_params_path, map_location=device)
        )
        max_epochs = 10
    else:
        # Part 3(c)：不加载权重，从零训练 75 Epochs
        max_epochs = 75

    # 4. 配置训练超参数
    tconf = trainer.TrainerConfig(
        max_epochs=max_epochs,
        batch_size=256,
        learning_rate=args.finetune_lr,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=200*len(pretrain_dataset)*block_size,
        num_workers=4,
        writer=writer
    )

    # 5. 创建 Trainer 并开始训练
    trainer_instance = trainer.Trainer(
        model, train_dataset, None, tconf
    )
    trainer_instance.train()

    # 6. 保存训练完成的模型参数
    torch.save(model.state_dict(), args.writing_params_path)

    ### END YOUR CODE ###

elif args.function == 'evaluate':
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.eval_corpus_path is not None
    model.load_state_dict(torch.load(args.reading_params_path))
    correct = 0
    total = 0
    with open(args.outputs_path, 'w', encoding='utf-8') as fout:
        predictions = []
        for line in tqdm(open(args.eval_corpus_path, encoding='utf-8')):
            x = line.split('\t')[0]
            x = x + '⁇'
            x = torch.tensor([pretrain_dataset.stoi[s] for s in x],
                             dtype=torch.long)[None,...].to(device)
            pred = utils.sample(model, x, 32, sample=False)[0]
            completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
            pred = completion.split('⁇')[1]
            predictions.append(pred)
            fout.write(pred + '\n')
        total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
    if total > 0:
        print(f'Correct: {correct} out of {total}: {correct/total*100}%')
    else:
        print(f'Predictions written to {args.outputs_path}; no targets provided')
