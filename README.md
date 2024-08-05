# Attention-Based Temporal Convolutional Networks and Reinforcement Learning for Supply Chain Delay Prediction and Inventory Optimization

**Paper**: [IEEE ICICML 2024](https://ieeexplore.ieee.org/document/10957909) | 6 Citations

## Abstract

This paper presents a novel framework integrating an Attention-Based Temporal Convolutional Network (ATCN) and Reinforcement Learning (RL) to address supply chain delay prediction and inventory optimization. The ATCN model leverages convolutional layers to capture long-term dependencies in time series data, while the attention mechanism enhances prediction accuracy under high demand volatility. The RL component optimizes inventory decisions by minimizing holding and shortage costs through multi-agent collaboration.

## Methods

- **Temporal Convolutional Network (TCN)** — Dilated causal convolutions for long-range temporal dependencies
- **Attention Mechanism** — Self-attention over TCN features for dynamic weighting
- **Reinforcement Learning** — DQN agent for inventory optimization (minimize holding + shortage costs)
- **Multi-Agent Collaboration** — Separate agents for different supply chain echelons

## Repository Structure

```
├── src/
│   ├── tcn.py                 # Temporal Convolutional Network
│   ├── attention_tcn.py       # TCN with self-attention (ATCN)
│   ├── rl_agent.py            # DQN agent for inventory optimization
│   ├── environment.py         # Supply chain simulation environment
│   ├── train_predictor.py     # Train delay prediction model
│   └── train_rl.py            # Train RL inventory agent
├── requirements.txt
└── README.md
```

## Results

| Model | MAE | MSE | R² | AUC |
|-------|-----|-----|-----|-----|
| LSTM | 2.34 | 8.12 | 0.82 | 0.87 |
| TCN | 2.01 | 6.45 | 0.86 | 0.89 |
| TCN + Attention | 1.78 | 5.21 | 0.89 | 0.92 |
| **ATCN + RL (Ours)** | **1.52** | **4.03** | **0.92** | **0.95** |

## Citation

```bibtex
@inproceedings{jin2024attention,
  title={Attention-Based Temporal Convolutional Networks and Reinforcement Learning for Supply Chain Delay Prediction and Inventory Optimization},
  author={Jin, Tian},
  booktitle={2024 International Conference on Image Processing, Computer Vision and Machine Learning (ICICML)},
  year={2024},
  organization={IEEE},
  doi={10.1109/ICICML63543.2024.10957909}
}
```

## Requirements

```
torch>=1.12
numpy
pandas
gymnasium
matplotlib
scikit-learn
```

## License

MIT
