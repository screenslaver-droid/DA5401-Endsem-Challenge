# Three-Signal Stacked Ensemble for Metric Learning

A sophisticated machine learning pipeline for evaluating text-metric quality using a three-signal decomposition approach with stacked ensemble modeling.

## Overview

This project addresses an out-of-distribution regression challenge where the training dataset was heavily skewed toward high scores (91% of samples scored 9.0 or 10.0), while the test set contained the full 0-10 range. The solution implements a three-signal expert system that decomposes quality assessment into independent dimensions, followed by ensemble meta-learning.

**Final Performance**: RMSE = 2.131

## Architecture

The pipeline operates in three stages:

### Stage 1: Three Independent Signals

The system trains three specialized "expert" models, each capturing a distinct quality dimension:

- **S1 (Anomaly Expert)**: Autoencoder trained exclusively on high-quality samples to detect anomalous metric-text pairings through reconstruction error analysis
- **S2 (Relevance Expert)**: Deep neural network binary classifier measuring semantic alignment between metrics and text using interaction features
- **S3 (Coherence Expert)**: Pre-trained multilingual Natural Language Inference model evaluating logical consistency between prompts and responses

### Stage 2: Synthetic Data Augmentation

To bridge the distribution gap, synthetic "failure" data is generated using a Gaussian Ladder strategy:

- **Partial Failures** (scores 4-8): Single-signal failures with noisy labeling for regularization
- **Total Failures** (scores 0-3): Multi-signal failures representing severe quality degradation

This augmentation proved critical for model generalization and required extensive hyperparameter tuning.

### Stage 3: Stacked Ensemble

A meta-learning framework combines the three signals:

- Multiple gradient boosting base models (XGBoost, LightGBM, Random Forest, GBM) generate 5-fold out-of-fold predictions
- Ridge regression meta-model learns optimal signal weights
- Extensive regularization at data, model, and ensemble levels prevents overfitting

## Key Technical Features

- **Embedding Strategy**: Google's embedding-gemma-300m with explicit [SEP] tokens for hierarchical context preservation
- **Feature Engineering**: Interaction features (element-wise multiplication and difference) capture semantic alignment patterns
- **NLI-Based Coherence**: Uses entailment probability as a theoretically grounded coherence measure
- **Multi-Level Regularization**: Dropout, batch normalization, early stopping, L1/L2 penalties, and noisy synthetic labels
- **Caching System**: Efficient computation management for embeddings and model predictions

## Model Components

### S1: Autoencoder Anomaly Detector

- Architecture: 3072 → 1024 → 512 → 48 (bottleneck) → 512 → 1024 → 3072
- Loss: Huber loss for outlier robustness
- Training: Semi-supervised on samples with score > 8.5

### S2: Metric-Matcher Classifier

- Input: Concatenated, interaction, and difference features (3072-dimensional)
- Architecture: 3072 → 512 → 256 → 128 → 1 (sigmoid output)
- Training: Binary classification with 2:1 negative sampling ratio

### S3: NLI Coherence Scorer

- Model: MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli
- Approach: Extracts entailment probability as coherence score
- Advantage: Explicitly models logical relationships rather than semantic similarity

### Ensemble Meta-Model

- Base Models: XGBoost, LightGBM, Random Forest, Gradient Boosting
- Stacking: RidgeCV with cross-validated alpha selection
- Features: 14 meta-features including signal interactions, polynomial terms, and statistical aggregations

## Results

The model successfully learned a smooth decision manifold rather than memorizing discrete rules, as evidenced by:

- Continuous color gradients in 3D signal space visualization
- Distinct stratification layers from Gaussian Ladder augmentation
- Strong discriminative signal distributions with appropriate bimodal characteristics
- Validation of AND-logic between signals (all must be high for high prediction)

## Implementation

Built with:
- TensorFlow/Keras for deep learning components
- Scikit-learn for ensemble methods and preprocessing
- Hugging Face Transformers for NLI models
- SentenceTransformers for text encoding
- XGBoost and LightGBM for gradient boosting
  
## Requirements
'''bash
pip install numpy pandas tensorflow tf-keras scikit-learn xgboost lightgbm sentence-transformers transformers torch
'''


## Usage

The main pipeline is executed through `main_enhanced_workflow()`, which handles:

1. Data loading and text encoding with caching
2. Sequential training of S1, S2, and S3 expert models
3. Synthetic data augmentation with configurable parameters
4. Meta-feature generation and ensemble training
5. Prediction generation and comprehensive visualization

## Key Insights

- **Balanced Overfitting**: Success required maintaining a delicate balance between a slightly overfit base model and a regularized ensemble
- **Augmentation Tuning**: Gaussian Ladder parameters proved highly sensitive and critical to final performance
- **Signal Independence**: Three-signal decomposition effectively captures orthogonal quality dimensions
- **Noisy Labels**: Strategic noise injection in synthetic data acted as effective regularization

## Author

Siddharth Nair (CE22B106)

## Course

DA5401 End-Semester Data Challenge



## Requirements

