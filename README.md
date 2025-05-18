# Deep Ensembles for Uncertainty Estimation in Regression

This implementation follows the paper:

Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles  
Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell  
https://arxiv.org/abs/1612.01474

## Overview

Each model in the ensemble predicts:
- Mean (μ): predicted value
- Variance (σ²): input-dependent aleatoric uncertainty

With multiple independently trained models:
- Epistemic uncertainty = variance of predicted means
- Aleatoric uncertainty = average of predicted variances

## Quick Usage

ensemble = DeepEnsemble(num_models=5)  
ensemble.fit(train_loader, val_loader, epochs=3)

x = torch.rand(1, 10)  
mean, epistemic, aleatoric = ensemble.inference(x)

## Output Example

Mean prediction: tensor([42.1])  
Epistemic uncertainty: tensor([0.62])  
Aleatoric uncertainty: tensor([1.04])

## Features

- Independent MLP regressors
- Gaussian NLL loss (nn.GaussianNLLLoss)
- Predicts both epistemic and aleatoric uncertainty
- Clean, minimal PyTorch implementation

## Extensions

- Add FGSM-based adversarial training
- Compare with MC Dropout
- Use on real-world datasets (UCI, PhysioNet)
- Convert model list to nn.ModuleList for checkpointing

## Installation

pip install torch scikit-learn

## License

MIT License
