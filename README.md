# 深度学习实践仓库

深度学习课程学习、实验与课设代码（文本分类方向为主）。

## 目录结构

```
├── experiments/   # 深度学习实验：中文新闻文本分类（cnews）等
├── study/         # 深度学习框架学习练习
└── keshe/         # 深度学习课设：中文情感分析（ChnSentiCorp + BERT 微调）、IMDB 影评分类
```

## 主要内容

- **experiments**：基于 Keras/PyTorch 的中文新闻文本分类实验（TextCNN / RNN / 注意力等模型）
- **keshe**：使用 Hugging Face Transformers 微调 `bert-base-chinese` 做 ChnSentiCorp 情感分析；LSTM / BiLSTM 对比实验（IMDB）

## 环境

```
Python 3.9+
torch / transformers / datasets / keras / scikit-learn / jieba
```

> 数据集（cnews / IMDB）因体积原因未上传，请按脚本内路径自行放置。
