# Bias-Type Neural Localization & Activation Steering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![TransformerLens](https://img.shields.io/badge/transformerlens-compatible-green.svg)](https://github.com/neelnanda-io/TransformerLens)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Mechanistic Analysis of Activation Steering for Bias Mitigation in GPT-2 Large**  
> *Discovering layer-specific bias processing and targeted intervention strategies*

## 🎯 Overview

This repository contains the complete implementation for **bias-type neural localization** research, demonstrating that different bias types are processed at dramatically different neural depths in language models. Our systematic analysis across 9 bias types and 3,016 examples reveals:

- **Surface-level biases** (physical appearance, nationality) emerge in **early layers (3-19)**
- **Complex social biases** (gender, religion, race) require **deeper processing (layers 21-31)**
- **Consistent 0.4 bias reduction** achieved through targeted activation steering
- **89% attention dominance** over residual stream processing

## 🚀 Key Features

### 🔍 **Comprehensive Bias Detection**
- Layer-by-layer probing across all 36 GPT-2 Large layers
- 72 hook points analysis (residual streams + attention outputs)
- Logistic regression classifiers with PCA dimensionality reduction
- Automated optimal layer identification for each bias type

### 🎯 **Activation Steering Framework**
- Real-time bias mitigation during text generation
- Steering vector computation from activation differences
- Configurable intervention strength (α = 0.75 default)
- Preserved text coherence and fluency

### 📊 **Advanced Analytics**
- Layer importance ranking and effectiveness analysis
- Bias-type specific performance metrics
- Comprehensive visualization suite
- Detailed statistical reporting

### 🛠️ **Production-Ready Pipeline**
- Memory-optimized for Google Colab (T4 GPU, 15GB VRAM)
- Robust error handling and progress tracking
- Model saving/loading with complete state preservation
- Extensive logging and debugging capabilities



## ⚡ Quick Start

### 1. Installation

```bash
# Clone repository
git clone 
cd Activation-Steering-for-Bias-Mitigation

# Install dependencies
pip install torch transformers transformer-lens scikit-learn pandas matplotlib seaborn tqdm
```

### 2. Download CrowS-Pairs Dataset

```bash
# Download from official repository
wget https://github.com/nyu-mll/crows-pairs/blob/master/data/crows_pairs_anonymized.csv
```



## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



⭐ **Star this repository** if you find it useful for your research!

*This work contributes to building safer, more equitable AI systems through mechanistic understanding and targeted intervention.*
