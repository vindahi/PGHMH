# PGHMH

## Prompt-Guided Hierarchical Multi-modal Hashing for Multi-modal Retrieval

## Overview
**PGHMH (Prompt-Guided Hierarchical Multi-modal Hashing)** is a novel approach designed to address the challenges of inefficient multi-modal fusion, imbalanced modality handling, and limited semantic guidance in multi-modal retrieval. By leveraging context-aware prompt generation and hierarchical alignment strategies, PGHMH achieves robust and efficient retrieval performance across multiple modalities.

### Key Highlights:
- **Context-Aware Prompt Generation:** Dynamically constructs semantic anchors based on category knowledge and sample characteristics. These anchors serve as critical guidance for attention-based feature fusion and cross-modal alignment.
- **Hierarchical Alignment Strategy:** Optimizes semantic consistency at multiple levels:
  1. Inter-modal alignment.
  2. Anchor-guided refinement.
  3. Structured binary encoding.
- **State-of-the-Art Performance:** PGHMH outperforms existing methods by an average of **9.98% in MAP** on benchmark datasets.

---

## Abstract
Multi-modal hashing aims to enable efficient cross-modal retrieval by learning compact binary codes that preserve semantic similarity. However, existing methods often suffer from ineffective fusion, imbalanced modality handling, and limited semantic guidance. To address these issues, we propose a novel method called **Prompt-Guided Hierarchical Multi-modal Hashing (PGHMH)**. Our method introduces a context-aware prompt generation mechanism that dynamically constructs semantic anchors based on category knowledge and sample characteristics. These anchors guide attention-based feature fusion and enhance cross-modal alignment. We further develop a hierarchical alignment strategy that jointly optimizes semantic consistency at multiple levels, from inter-modal alignment to anchor-guided refinement, and finally to structured binary encoding. Experiments on benchmark datasets show that PGHMH outperforms state-of-the-art methods by an average of **9.98% in MAP**.

---

## Datasets
PGHMH is evaluated on three widely-used benchmark datasets: **MIRFlickr**, **NUS-WIDE**, and **MS COCO**. The datasets can be downloaded from **[Baidu Drive](https://pan.baidu.com/share/init?surl=ZyDTR2IEHlY4xIdLgxtaVA) (Password: kdq7)**. The directory structure and dataset details are as follows:

### Dataset Structure:
The datasets are structured as follows:
```bash
dataset
├── coco
│ ├── caption.mat
│ ├── index.mat
│ └── label.mat
├── flickr25k
│ ├── caption.mat
│ ├── index.mat
│ └── label.mat
├── nuswide
│ ├── caption.txt
│ ├── index.mat
│ └── label.mat
```

### Dataset Details:
| Dataset    | Train Samples | Query Samples | Retrieval Samples | Categories |
|------------|---------------|---------------|-------------------|------------|
| MIRFlickr  | 10,000        | 2,000         | 22,581            | 24         |
| NUS-WIDE   | 10,000        | 2,000         | 190,779           | 21         |
| MS COCO    | 10,000        | 5,000         | 117,218           | 80         |

---

## Experimental Setup
### Hardware and Software
- **Python Version:** 3.11.18 
- **PyTorch Version:** 2.3.1  
- **GPU:** NVIDIA RTX 3090  

### Key Files and Directories:
1. **Hash Codes Directory:**
   - Hash codes for different bit lengths (**16, 32, 64, 128**) are stored in the `flickrhashcode` directory for result validation.
2. **Model Checkpoints:**
   - Trained models for each dataset are saved in the `checkpoint` directory for testing and evaluation.

---

## Usage
To reproduce the results, follow these steps:

### Step 1: Train the Model
Run the training script for the desired dataset. For example, to train on the **MIRFlickr** dataset, execute:
```bash
bash flickr.sh
```

### Step 2: Evaluate the Model
The trained hash codes and models can be used to validate performance on retrieval tasks. Hash codes are stored in the flickrhashcode directory for further evaluation.

## Results
PGHMH achieves state-of-the-art performance across multiple datasets, outperforming existing methods by an average of 9.98% in MAP. Detailed results are as follows:
| Dataset    | MAP Improvement (%) | 
|------------|---------------|
| MIRFlickr  | +7.43       | 
| NUS-WIDE   | +19.12        | 
| MS COCO    | +3.17       |

These results demonstrate the effectiveness and robustness of PGHMH under various scenarios.
