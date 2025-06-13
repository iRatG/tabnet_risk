# TabNet Credit Risk Model

This repository implements a production-grade credit risk modeling pipeline using TabNet and PyTorch.

## Structure

- `modules/data_preparation/`: Data loading, target generation, aggregation
- `modules/modeling/`: TabNet architecture and training
- `modules/evaluation/`: Metrics and visualizations
- `scripts/`: Entry points for full training pipelines
- `data/`: Place to store raw/synthetic data

## Getting Started

```bash
pip install -r requirements.txt
python scripts/train_model.py
```
