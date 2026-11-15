# Stanford CS224n — Spring 2024 Assignments

本仓库用于记录学习 Stanford CS224n: Natural Language Processing with Deep Learning（Spring 2024）的过程，包括课程作业、代码实现和实验分析。

## Assignments

- **Assignment 1: Exploring Word Vectors** — 已完成
- **Assignment 2: Neural Dependency Parsing** — 已完成 1(a)–(c)、2(a)–(b)、3(a)–(e)；数学推导记录于个人笔记，未上传 GitHub
- **Assignment 3: Neural Machine Translation** — 已完成 Part (a)；Part (b) 未完成

## Assignment 2: Neural Dependency Parsing

### Implementation

- 实现 Transition-Based Dependency Parsing，包括 SHIFT、LEFT-ARC 和 RIGHT-ARC。
- 实现 Mini-batch Parsing。
- 使用 PyTorch 构建基于 Embedding 和 MLP 的神经网络依存句法分析器。
- 实现 ReLU、Dropout 和前向传播。
- 使用 Adam 优化器与 CrossEntropyLoss 完成模型训练。
- 使用 UAS（Unlabeled Attachment Score）评估模型性能。

### Experimental Results

| Metric | Result |
|---|---:|
| Training Epochs | 10 |
| Batch Size | 1024 |
| Learning Rate | 0.0005 |
| Optimizer | Adam |
| Final Train Loss | 0.0661 |
| Dev UAS | 88.76% |
| Test UAS | 88.98% |

### Key Learnings

- PyTorch Tensor indexing、reshape 与矩阵乘法
- `nn.Module`、`nn.Parameter` 与模型前向传播
- Mini-batch Training 与 CrossEntropyLoss
- Automatic Differentiation 与 Backpropagation
- Adam Optimizer 与模型参数更新
- 模型训练、验证、测试及最佳权重保存

## Assignment 3: Neural Machine Translation

### Implementation

- 实现 Sentence Padding，统一 Mini-batch 中句子的长度。
- 基于 PyTorch 和课程提供的框架，实现 Seq2Seq Neural Machine Translation。
- 使用 Bidirectional LSTM Encoder 和 Unidirectional LSTM Decoder。
- 实现 Multiplicative Attention 和 Input Feeding。
- 使用 1D Convolution 提取输入序列的局部特征。
- 使用 Adam 优化器训练模型，并通过 Perplexity 进行验证。
- 使用 Beam Search 解码，并通过 Corpus BLEU 评估翻译质量。

### Experimental Results

| Metric | Result |
|---|---:|
| Task | Chinese–English Translation |
| Training Platform | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Framework | PyTorch |
| Embedding Size | 1024 |
| Hidden Size | 768 |
| Optimizer | Adam |
| Evaluation Metric | Corpus BLEU |
| Test BLEU | **20.21** |
| Assignment Requirement | BLEU > 18 |
| Result | Passed |

### Key Learnings

- Sequence-to-Sequence Learning
- Bidirectional LSTM Encoder–Decoder
- Attention Mechanism 与 Input Feeding
- 1D Convolution 与局部特征提取
- Beam Search Decoding
- Perplexity (PPL) 与 BLEU
- Checkpoint 保存与恢复
- Learning Rate Decay 与 Early Stopping
- GPU Training 与模型性能评估

### Notes

- 已完成 Assignment 3 Part (a)，Part (b) 未完成。
- 模型训练和测试在 Google Colab 上完成，最终测试集 Corpus BLEU 为 **20.21**，达到作业要求（BLEU > 18）。
- 本仓库保留本地 PyCharm 中的代码实现。Colab 中针对模型配置、训练环境和 Checkpoint 恢复的部分修改未同步至本地，因此本地代码与实际训练配置并不完全一致。
- 未上传训练模型权重、优化器状态文件及 Colab Notebook。

## Tech Stack

- Python
- NumPy
- PyTorch
- Gensim
- Scikit-learn
- Jupyter Notebook
- Google Colab

## References

- [CS224n Official Website](https://web.stanford.edu/class/cs224n/)

## Note

本仓库为个人学习记录，非 Stanford 官方课程仓库。

Assignment 2 和 Assignment 3 的实现基于课程提供的作业框架。实验结果来自个人运行，其中 Assignment 3 的模型训练和测试在 Google Colab 上完成。