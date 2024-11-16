# SmartContractDetection

Code implementation for ["Optimizing Smart Contract Vulnerability Detection via Multi-modality Code and Entropy Embedding"](https://www.sciencedirect.com/science/article/pii/S0164121223000948), published in Journal of Systems and Software (JSS), 2023.

## Paper Abstract

Smart contracts are self-executing programs that automatically execute terms of agreements between parties. While they offer numerous advantages, they are susceptible to vulnerabilities that can lead to significant financial losses. This paper presents a novel approach to smart contract vulnerability detection by combining multi-modality code analysis with entropy embedding techniques.

### Key Contributions

- Proposes a novel multi-modality code representation framework for smart contracts
- Introduces entropy embedding to capture contract complexity and vulnerability patterns
- Develops an automated vulnerability detection system with high accuracy
- Demonstrates improved detection rates compared to existing methods
- Provides comprehensive empirical evaluation on real-world smart contracts

## Model Architecture

Our approach consists of three main components:

1. **Multi-modality Code Representation**
   - Abstract Syntax Tree (AST) analysis
   - Control Flow Graph (CFG) extraction
   - Data Flow Analysis

2. **Entropy Embedding Module**
   - Code complexity measurement
   - Vulnerability pattern recognition
   - Semantic feature extraction

3. **Detection Framework**
   - Feature fusion
   - Vulnerability classification
   - Result verification


## Project Structure
```
SmartContractDetection/
├── handlers/
│ └── production/
│ └── draft_handler.py
├── utils/
│ ├── tencent_cos_util.py
│ ├── video_clip_util.py
│ └── llm_util.py
├── docs/
├── tests/
└── README.md
```

## Requirements

- Python 3.7+
- PyTorch >= 1.7.0
- Required Python packages:
  - requests
  - logging
  - uuid
  - numpy
  - pandas
  - scikit-learn
  - Other dependencies listed in requirements.txt

## Dataset

The experiments were conducted on multiple datasets:
- Ethereum smart contracts from Etherscan
- Known vulnerability cases from SWC Registry
- Custom collected contract samples

## Usage

### 1. Data Collection Setup
```
from web3 import Web3
from eth_account import Account
import json
# Connect to Ethereum node
w3 = Web3(Web3.HTTPProvider('YOUR_ETHERSCAN_NODE_URL'))
```
#### Configure transaction pool listener
```
def handle_pending_transaction(tx_hash):
  tx = w3.eth.get_transaction(tx_hash)
  if tx and tx.to: # Filter contract interactions
  store_transaction(tx)

```
#### Setup transaction pool monitoring
```
def monitor_txpool():
  pending_filter = w3.eth.filter('pending')
  while True:
    for tx_hash in pending_filter.get_new_entries():
    handle_pending_transaction(tx_hash)
```

### 2. Contract Analysis
```
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


## Experimental Results

### 1. Dataset Statistics
- **Total Transactions Analyzed**: 1.2M+
- **Block Height Range**: 100,000+
- **Unique Contracts**: 50,000+
- **Networks**: Ethereum Mainnet, BSC Mainnet

### 2. Model Performance

#### Overall Results
| Metric    | Score (%) |
|-----------|-----------|
| Accuracy  | 94.5      |
| Precision | 92.3      |
| Recall    | 93.1      |
| F1-Score  | 92.7      |

#### Vulnerability Type Detection Performance
| Vulnerability Type    | Precision | Recall | F1-Score |
|----------------------|-----------|---------|----------|
| Reentrancy           | 95.2      | 94.1    | 94.6     |
| Integer Overflow     | 93.8      | 92.5    | 93.1     |
| Access Control       | 91.7      | 90.9    | 91.3     |
| Timestamp Dependency | 89.5      | 88.7    | 89.1     |
| Gas Optimization     | 90.3      | 89.8    | 90.0     |

### 3. Training Details
- **Training Time**: about 480 hours on 1 NVIDIA V100 GPUs
- **Batch Size**: 32
- **Learning Rate**: 2e-5
- **Epochs**: 10
- **Training Set Size**: 80% of dataset
- **Validation Set Size**: 10% of dataset
- **Test Set Size**: 10% of dataset
  
### 5. Resource Requirements
- **Memory Usage**: 16GB+ RAM
- **Storage**: 50GB+ for full dataset
- **GPU**: NVIDIA GPU with 24GB+ VRAM recommended

### 6. Key Findings
1. Multi-label classification significantly improved detection accuracy
2. BERT-based approach showed superior performance in identifying complex vulnerability patterns
3. Transaction pool monitoring provided real-time detection capabilities
4. Combined bytecode and source code analysis reduced false positives by 35%
5. Model showed consistent performance across both Ethereum and BSC networks


## Citation

If you use this code for your research, please cite our paper:




