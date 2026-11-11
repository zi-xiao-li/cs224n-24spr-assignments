# Stanford CS224n — Spring 2024 Assignments

本仓库用于记录学习 Stanford CS224n: Natural Language Processing with Deep Learning（Spring 2024）的过程，包括课程作业、代码实现和实验分析。

## Assignments

- **Assignment 1: Exploring Word Vectors** — 已完成
- **Assignment 2: Neural Dependency Parsing** — 核心编程部分已完成，数学推导部分未上传

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

## Tech Stack

- Python
- NumPy
- PyTorch
- Gensim
- Scikit-learn
- Jupyter Notebook

## References

- [CS224n Official Website](https://web.stanford.edu/class/cs224n/)

## Note

本仓库为个人学习记录，非 Stanford 官方课程仓库。Assignment 2 的实现基于课程提供的作业框架，实验结果来自个人运行。

