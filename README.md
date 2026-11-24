
# Stanford CS224n — Spring 2024 Assignments

本仓库用于记录学习 Stanford CS224n: Natural Language Processing with Deep Learning（Spring 2024）的过程，包括课程作业、代码实现和实验分析。

## Assignments

- **Assignment 1: Exploring Word Vectors** — 已完成
- **Assignment 2: Neural Dependency Parsing** — 已完成 1(a)–(c)、2(a)–(b)、3(a)–(e)；数学推导记录于个人笔记，未上传
- **Assignment 3: Neural Machine Translation** — 已完成 Part (a) 及模型训练与评估
- **Assignment 4: Self-Attention, Transformers, and Pretraining** — 已完成

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

- 模型训练和测试在 Google Colab 上完成，最终测试集 Corpus BLEU 为 **20.21**，达到作业要求（BLEU > 18）。
- 本仓库保留本地 PyCharm 中的代码实现。Colab 中针对模型配置、训练环境和 Checkpoint 恢复的部分修改未同步至本地，因此本地代码与实际训练配置并不完全一致。
- 未上传训练模型权重、优化器状态文件及 Colab Notebook。

## Assignment 4: Self-Attention, Transformers, and Pretraining

### Implementation

- 基于课程提供的 minGPT 框架，实现并训练 Character-level GPT 语言模型。
- 实现 Span Corruption 数据集，包括随机截断、Span Masking、特殊字符处理及输入与目标序列构建。
- 使用 Wikipedia 文本语料进行自监督预训练，并在人物出生地预测任务上进行 Fine-tuning。
- 实现 Rotary Position Embeddings（RoPE），包括旋转角度预计算、复数形式的旋转操作，以及在 Multi-head Causal Self-Attention 中对 Query 和 Key 应用 RoPE。
- 对比无预训练 Vanilla GPT、预训练 Vanilla GPT 和 RoPE GPT 的预测性能。
- 实现 London Baseline，并在开发集上评估模型准确率。

### Experimental Results

| Model | Dev Accuracy | Correct Predictions |
|---|---:|---:|
| Vanilla GPT (Without Pretraining) | 1.8% | 9/500 |
| London Baseline | 5.0% | 25/500 |
| Vanilla GPT (Pretrained + Fine-tuned) | 20.8% | 104/500 |
| **RoPE GPT (Pretrained + Fine-tuned)** | **28.6%** | **143/500** |

### Training Configuration

| Metric | Result |
|---|---|
| Framework | PyTorch |
| Training Platform | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Pretraining Corpus | Wikipedia-derived Text |
| Fine-tuning Task | Birthplace Prediction |
| Context Length | 128 |
| Transformer Layers | 4 |
| Attention Heads | 8 |
| Embedding Dimension | 256 |
| Pretraining Epochs | 650 |
| Fine-tuning Epochs (Pretrained Models) | 10 |
| Evaluation Metric | Exact-match Accuracy |

### Key Learnings

- GPT Architecture 与 Causal Self-Attention
- Multi-head Attention 与 Causal Masking
- Character-level Autoregressive Language Modeling
- Span Corruption 与 Self-supervised Pretraining
- Pretraining 与 Fine-tuning 的区别
- Rotary Position Embeddings（RoPE）及相对位置信息编码
- PyTorch Tensor Reshape、Broadcasting 与 Complex Tensor Operations
- GPU Training、Checkpoint 保存与模型评估
- Pretrained Knowledge、Hallucination 与模型预测可靠性

### Notes

- Vanilla GPT 和 RoPE GPT 均完成 650 epochs 预训练及 10 epochs 微调。
- RoPE GPT 在开发集上取得 **28.6%** 的准确率，相比预训练 Vanilla GPT 的 **20.8%** 提高 **7.8 个百分点**。
- 已生成开发集和测试集预测文件。测试集真实标签未公开，因此不报告测试集准确率。
- 模型训练、微调和评估在 Google Colab 上完成。
- 预训练权重、微调权重及预测结果已备份至 Google Drive，未上传至 GitHub。
- 本仓库主要保留 A4 的 Python 源代码实现。

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