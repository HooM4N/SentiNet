# SentiNet
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-red?logo=pytorch)
![Transformers](https://img.shields.io/badge/Transformers-4.57.0-orange?logo=huggingface)
![Tokenizers](https://img.shields.io/badge/Tokenizers-0.22.1-lightgrey?logo=huggingface)
![DeBERTa v3](https://img.shields.io/badge/DeBERTa-v3-green?logo=microsoft)
![Dataset](https://img.shields.io/badge/Dataset-IMDb-red)
![Task](https://img.shields.io/badge/Task-Sentiment_Analysis-blue)
![License](https://img.shields.io/badge/License-MIT-blue)

## 🧠 Introduction

**SentiNet** aims to tackle the *“Hello World”* of NLP — **IMDb movie review sentiment analysis**.
Although the dataset appears simple, it’s far from trivial. Achieving high accuracy requires robust natural language understanding to handle **sarcasm**, **sentiment flips**, and **subtle linguistic cues**.

In this project, I explored several approaches to this task:

| Approach                                | Description                                                                                                                                                                                                              |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Fine-tuned Transformer (DeBERTa-v3)** | Fine-tuned **Microsoft’s DeBERTa-v3** on IMDb reviews with a custom projection head and `[CLS]` pooling for classification. Achieves strong contextual understanding and robustness to sarcasm and sentiment flips. |
| **BiGRU + Pretrained GloVe Embeddings**            | A **Bidirectional GRU** network initialized with **GloVe embeddings**, capturing sequential dependencies and semantic similarity from pretrained word vectors.                                                           |
| **BiGRU + WordPiece Tokenizer**         | Another **BiGRU** model, but trained using a **custom WordPiece tokenizer** to better handle rare and out-of-vocabulary words.                                                                                           |
| **Classic ML Baselines**                | Traditional models (Logistic Regression, Naïve Bayes, Gradient Boosting, Bagging) trained on TF-IDF features for comparison and benchmarking.                                                                            |

📓 **Comprehensive Jupyter Notebook**: [https://github.com/Hoom4n/SentiNet/blob/main/SentiNet.ipynb](https://github.com/Hoom4n/SentiNet/blob/main/SentiNet.ipynb)

🚀 **Try Online Demo on Hugging Face 🤗**: (TODO)

🗂️ **Transformer-Based Model on Hugging Face**: (TODO)

## 📊 Results

I evaluated models both quantitatively (via **F1 score**) and qualitatively on examples featuring linguistic nuances such as sarcasm, mixed sentiment, and negation.

### 🧩 **Quantitative Evaluation**

| Model                        | Train F1 | Val F1    | Test F1   |
| ---------------------------- | -------- | --------- | --------- |
| Logistic Regression + TF-IDF | 0.939    | 0.897     | 0.886     |
| BiGRU + WordPiece Tokenizer  | 0.928    | 0.854     | 0.850     |
| BiGRU + Pretrained GloVe Embeddings     | 0.921    | 0.881     | 0.866     |
| Transformer (DeBERTa-v3)     | —        | **0.948** | **0.954** |



### 🧩 **Qualitative NLU Evaluation**

The table below summarizes how each model handled five linguistically challenging samples — including sarcasm, shifting sentiment, and negation-based flips.
Confidence values are shown in parentheses.

| Text                                                                                 | Actual Sentiment | Challenge Type             | Logistic Reg.         | BiGRU + GloVe         | BiGRU + WordPiece     | Transformer (DeBERTa-v3) |
| ------------------------------------------------------------------------------------ | ---------------- | -------------------------- | --------------------- | --------------------- | --------------------- | ------------------------ |
| *The movie was short, simple, and absolutely wonderful.*                             | Positive         | Straightforward sentiment  | ✅ **Positive (0.99)** | ✅ **Positive (0.97)** | ✅ **Positive (0.92)** | ✅ **Positive (1.00)**    |
| *The first half was boring and predictable, but the ending completely blew me away.* | Positive         | Shifting tone (neg→pos)    | ❌ **Negative (0.89)** | ❌ **Negative (0.93)** | ❌ **Negative (0.88)** | ✅ **Positive (0.98)**    |
| *Yeah, sure, this was the “best” film ever... if you enjoy watching paint dry.*      | Negative         | Sarcasm / irony            | ❌ **Positive (0.72)** | ❌ **Positive (0.95)** | ✅ **Negative (0.59)** | ✅ **Negative (0.88)**    |
| *The acting was decent, but the script was weak and the pacing dragged.*             | Negative         | Mixed but overall negative | ✅ **Negative (0.98)** | ✅ **Negative (0.91)** | ✅ **Negative (0.79)** | ✅ **Negative (0.98)**    |
| *I didn’t expect much, yet it turned out surprisingly good.*                         | Positive         | Negation & contrast flip   | ✅ **Positive (0.84)** | ✅ **Positive (0.93)** | ✅ **Positive (0.93)** | ✅ **Positive (0.98)**    |

