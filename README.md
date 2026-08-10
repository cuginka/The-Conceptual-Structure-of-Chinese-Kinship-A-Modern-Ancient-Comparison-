# The Conceptual Structure of Chinese Kinship: A Modern-Ancient Comparison

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Manuscript_in_Preparation-green)]()

> **Research Project** | **Advisor:** Prof. Duan Jipeng | **Ningbo University**

## 📖 Overview
This repository contains the computational pipeline and analysis code for my independent research project: *"Mapping the Semantic Drift of Chinese Kinship Terms across 3000 Years."*

This project bridges **Computational Linguistics** and **Social Cognition** by comparing human mental representations with high-dimensional embeddings from Large Language Models (LLMs). We investigate how the semantic geometry of kinship terms (e.g., *Father*, *Maternal Uncle*) has evolved from ancient texts to modern usage.

## 🚀 Key Features & Pipeline
The analysis framework integrates **Representational Similarity Analysis (RSA)** to benchmark behavioral data against artificial neural networks.

### 1. Behavioral Data Acquisition
- Collected similarity ratings and kinship structure data from **N=160** human participants.
- Constructed human-based **Representational Dissimilarity Matrices (RDMs)** based on cognitive dimensions (generation, gender, lineage).

### 2. Computational Modeling (LLMs)
- **Context Generation:** Automated prompt engineering using **DeepSeek-V3**.
- **Embedding Extraction:** Extracted contextualized embeddings from:
  - **Modern Models:** `Chinese-RoBERTa-wwm`, `BERT-base-chinese`
  - **Ancient Models:** `SikuBERT` (trained on Siku Quanshu) / `Guwen-BERT`
- **Dimensionality Reduction:** Applied **t-SNE** and **MDS** to visualize the semantic manifolds.

### 3. Neural-AI Alignment (RSA)
- Calculated the Spearman correlation between Human RDMs and Model RDMs.
- Quantified the "alignment score" to measure how well LLMs capture the anthropological structure of kinship.
