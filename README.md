# Deception_Detection# Deception Detection in Diplomacy

This repository contains code, notebooks, and data for reproducing experiments on detecting deception in the board game Diplomacy. We propose and compare two complementary models:

- **HiS-Attention**: A transformer‑based multimodal fusion model integrating message content, game state features, and conversational history.
- **LieDetectorGAT**: A graph attention network explicitly modeling player interaction graphs with linguistic deception cues and power dynamics.

---

## Table of Contents
1. [Authors](#authors)
2. [Abstract](#abstract)
3. [Repository Structure](#repository-structure)
4. [Requirements](#requirements)
5. [Data Setup](#data-setup)
6. [Model Weights](#model-weights)
7. [Usage](#usage)
8. [Results](#results)
9. [Error Analysis](#error-analysis)
10. [Future Work](#future-work)
11. [License](#license)

---

## Authors
- **Dasari Sai Harsh** (2022144) — dasari22144@iiitd.ac.in  
- **Avi Sharma** (2022119) — avi22119@iiitd.ac.in  
- **Parth Sandeep Rastogi** (2022352) — parth22352@iiitd.ac.in

## Abstract
> Deception detection is a critical yet underexplored task in multi-agent strategy games like Diplomacy, where players often engage in calculated misdirection. Building upon prior work, we propose two models: (1) HiS‑Attention, a transformer‑based multimodal fusion architecture; and (2) LieDetectorGAT, a graph attention network modeling inter-player interactions enriched with linguistic cues and power dynamics. Through detailed error and speaker‑level analyses, we demonstrate the importance of contextual modeling.

## Repository Structure
```
├── data/                        # Raw and processed datasets
│── gat_model.pt             # LieDetectorGAT model weights
│── his-attention.ipynb      # Train and evaluate HiS‑Attention
│── gatmodel.ipynb           # Train and evaluate LieDetectorGAT
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Requirements
- Python 3.8+
- PyTorch
- PyTorch Geometric
- Transformers (Hugging Face)
- NumPy, Pandas, scikit-learn
- Jupyter Notebook or JupyterLab

Install via:
```bash
pip install -r requirements.txt
```

## Data Setup
1. **HiS‑Attention embeddings**: Download from Kaggle:
   ```bash
   kaggle datasets download -d harsh99429/attention-tranformer-all-embed
   unzip attention-tranformer-all-embed.zip -d data/embeddings
   ```
2. **Diplomacy Deception dataset**: Place raw JSON/CSV files under `data/raw/` if not included.

## Model Weights
- Copy `gat_model.pt` to `models/` before running `gatmodel.ipynb`.

## Usage
1. Launch Jupyter:
   ```bash
   jupyter lab
   ```
2. Open and run cells in:
   - `his-attention.ipynb` to train and evaluate the HiS‑Attention model.
   - `gatmodel.ipynb` to train and evaluate the LieDetectorGAT model.

Each notebook contains detailed instructions on data loading, preprocessing, training hyperparameters, and evaluation metrics (Macro F1, Lie F1, confusion matrices).

## Results
| Model                          | Macro F1 | Lie F1 |
|--------------------------------|----------|--------|
| NoContext + ConcatFusion       | 0.51     | 0.00   |
| Last5 + BiLSTM + MeanAttention | 0.538    | 0.137  |
| HiS‑Attention (Ours)           | 0.5738   | 0.2198 |
| LieDetectorGAT (Ours)          | 0.580    | 0.266  |

Confusion matrices and detailed per‑class metrics are generated within respective notebooks.

## Error Analysis
Find comprehensive quantitative and qualitative error analyses in Sections 6 of the project report. The notebooks also log misclassified examples and speaker‑wise error rates.

## Future Work
- Incorporate reinforcement learning or agent‑based simulation for dynamic strategy modeling.
- Enhance linguistic cue extraction using LIWC for richer psychological features.
- Fine‑tune large language models specifically for deception understanding.

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.

