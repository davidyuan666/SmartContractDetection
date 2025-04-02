# SmartContractDetection 🔍

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.jss.2023.111699-purple)](https://doi.org/10.1016/j.jss.2023.111699)
[![Paper PDF](https://img.shields.io/badge/📄_Full_Paper-ScienceDirect-red)](https://www.sciencedirect.com/science/article/pii/S0164121223000948)

## 📑 Publication Information
**Title**: Optimizing Smart Contract Vulnerability Detection via Multi-modality Code and Entropy Embedding  
**Journal**: [Journal of Systems and Software](https://www.sciencedirect.com/journal/journal-of-systems-and-software) (JSS)  
**Year**: 2023  
**DOI**: [10.1016/j.jss.2023.111699](https://doi.org/10.1016/j.jss.2023.111699)  
**Full Text**: [ScienceDirect Article](https://www.sciencedirect.com/science/article/pii/S0164121223000948)  

## 🧠 Abstract
> Smart contracts are self-executing programs that automatically execute terms of agreements between parties. While they offer numerous advantages, they are susceptible to vulnerabilities that can lead to significant financial losses. This paper presents a novel approach to smart contract vulnerability detection by combining:
> - ​**Multi-modality code analysis** 
> - ​**Entropy embedding techniques**
>
> Our method achieves state-of-the-art performance in detecting critical vulnerabilities.

## ✨ Key Features
| Feature | Description |
|---------|-------------|
| 🔗 Multi-modality Analysis | Combines bytecode and source code features |
| ⚡ Entropy Embedding | Captures code behavior patterns |
| 📊 High Accuracy | Outperforms existing vulnerability detectors |
| 💻 EVM Support | Compatible with Ethereum smart contracts |

## 🏗️ Model Architecture

Our approach consists of three main components:

### 1. 🔄 Multi-modality Code Representation
- 🌳 Abstract Syntax Tree (AST) analysis
- 📊 Control Flow Graph (CFG) extraction
- 📈 Data Flow Analysis

### 2. 🧮 Entropy Embedding Module
- 📏 Code complexity measurement
- 🔍 Vulnerability pattern recognition
- 🎯 Semantic feature extraction

### 3. 🛡️ Detection Framework
- 🔗 Feature fusion
- 🎯 Vulnerability classification
- ✅ Result verification

## 📋 Requirements

### Core Dependencies
- Python 3.7+
- PyTorch >= 1.7.0

### Python Packages
```python
requests==2.31.0
logging==0.5.1.2
uuid==1.30
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
# See requirements.txt for complete list
```

## 📊 Dataset Sources

- 💹 Ethereum smart contracts (Etherscan & BSCscan)
- 🔒 Known vulnerability cases (SWC Registry)
- 📚 Custom collected contract samples

## 💻 Usage Guide

### 1. Data Collection Setup
```python
from web3 import Web3
from eth_account import Account
import json

# Connect to Ethereum node
w3 = Web3(Web3.HTTPProvider('YOUR_ETHERSCAN_NODE_URL'))
```

### 2. Transaction Pool Management
```python
def handle_pending_transaction(tx_hash):
    tx = w3.eth.get_transaction(tx_hash)
    if tx and tx.to:  # Filter contract interactions
        store_transaction(tx)

def monitor_txpool():
    pending_filter = w3.eth.filter('pending')
    while True:
        for tx_hash in pending_filter.get_new_entries():
            handle_pending_transaction(tx_hash)
```

### 3. Contract Analysis
```python
from solidity_parser import parser

def analyze_contract(source_code, bytecode):
    # Parse source code
    ast = parser.parse(source_code)
    
    # Extract features
    features = {
        'ast_features': extract_ast_features(ast),
        'bytecode_features': analyze_bytecode(bytecode),
        'transaction_patterns': get_transaction_patterns()
    }
    return features
```

## 📈 Experimental Results

### Dataset Overview
| Metric | Value |
|:------:|:-----:|
| Total Transactions | 1.2M+ |
| Block Height Range | 100,000+ |
| Unique Contracts | 50,000+ |
| Networks | Ethereum & BSC Mainnet |

### Performance Metrics

#### Overall Results
| Metric | Score (%) |
|:------:|:---------:|
| Accuracy | 94.5 |
| Precision | 92.3 |
| Recall | 93.1 |
| F1-Score | 92.7 |

#### Vulnerability Detection Performance
| Vulnerability Type | Precision | Recall | F1-Score |
|:-----------------:|:---------:|:-------:|:--------:|
| Reentrancy | 95.2 | 94.1 | 94.6 |
| Integer Overflow | 93.8 | 92.5 | 93.1 |
| Access Control | 91.7 | 90.9 | 91.3 |
| Timestamp Dependency | 89.5 | 88.7 | 89.1 |
| Gas Optimization | 90.3 | 89.8 | 90.0 |

### Training Configuration
| Parameter | Value |
|:---------:|:-----:|
| Training Time | 480 hours |
| GPU | NVIDIA V100 |
| Batch Size | 32 |
| Learning Rate | 2e-5 |
| Epochs | 10 |

### Data Split
- 🔵 Training Set: 80%
- 🟡 Validation Set: 10%
- 🔴 Test Set: 10%

### System Requirements
| Resource | Minimum Requirement |
|:--------:|:-----------------:|
| RAM | 16GB+ |
| Storage | 50GB+ |
| GPU VRAM | 24GB+ |

### 🎯 Key Findings

1. Multi-label classification achieved superior detection accuracy
2. BERT-based approach excelled in complex vulnerability pattern identification
3. Real-time detection enabled through transaction pool monitoring
4. Combined analysis reduced false positives by 35%
5. Consistent performance across Ethereum and BSC networks

## 📝 Citation
```bibtex
@article{SmartContractDetection2023,
  title = {Optimizing Smart Contract Vulnerability Detection via Multi-modality Code and Entropy Embedding},
  journal = {Journal of Systems and Software},
  volume = {195},
  pages = {111699},
  year = {2023},
  doi = {10.1016/j.jss.2023.111699},
  url = {https://www.sciencedirect.com/science/article/pii/S0164121223000948}
}




