"""
Enhanced Model (v3) - NLI-Based Coherence Scoring System
Author: Siddharth Nair

Description:
    A multi-signal scoring system that evaluates text-metric pairs using:
    - S1: Autoencoder-based anomaly detection
    - S2: Trained metric-matcher classifier
    - S3: NLI-based prompt-response coherence
    
    The system uses ensemble meta-models for final score prediction.
"""

import numpy as np
import pandas as pd
import json
import os
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Machine Learning Libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
import xgboost as xgb
import lightgbm as lgb

# Deep Learning Libraries
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers
from tf_keras.models import Model

# NLP Libraries
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("ENHANCED MODEL (v3) - NLI-BASED COHERENCE")
print("=" * 70)


# ============================================================================
# DATA LOADING & ENCODING
# ============================================================================

def load_data():
    """
    Load training and test data along with metric embeddings.
    
    Returns:
        tuple: (metric_to_embedding dict, train_data list, test_data list)
    """
    # Load metric embeddings and names
    metric_embeddings = np.load(r'C:\Users\Siddharth Nair\Downloads\metric_name_embeddings.npy')
    with open(r'C:\Users\Siddharth Nair\Downloads\metric_names.json', 'r', encoding="utf-8") as f:
        metric_names = json.load(f)
    
    # Create mapping from metric name to embedding
    metric_to_embedding = {name: metric_embeddings[i] for i, name in enumerate(metric_names)}
    
    # Load training and test datasets
    with open(r'C:\Users\Siddharth Nair\Downloads\train_data.json', 'r', encoding="utf-8") as f:
        train_data = json.load(f)
    with open(r'C:\Users\Siddharth Nair\Downloads\test_data.json', 'r', encoding="utf-8") as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(train_data)} train, {len(test_data)} test samples.")
    return metric_to_embedding, train_data, test_data


def encode_text_data(data, model_name='google/embeddinggemma-300m', cache_file_path=None):
    """
    Encode text data using SentenceTransformer embeddings.
    
    Args:
        data: List of samples containing prompts, system_prompts, and responses
        model_name: Name of the SentenceTransformer model to use
        cache_file_path: Optional path to cache/load embeddings
        
    Returns:
        np.ndarray: Text embeddings of shape (n_samples, embedding_dim)
    """
    # Load from cache if available
    if cache_file_path and os.path.exists(cache_file_path):
        print(f"Loading cached embeddings from {cache_file_path}...")
        return np.load(cache_file_path)
    
    # Initialize encoder
    print(f"Loading SentenceTransformer: {model_name}")
    encoder = SentenceTransformer(model_name, trust_remote_code=True)
    
    # Combine text fields with [SEP] token
    texts = []
    for sample in data:
        prompt = sample.get('prompt', '')
        system_prompt = sample.get('system_prompt', '')
        response = sample.get('expected_response', sample.get('response', ''))
        combined_text = f"{system_prompt} [SEP] {prompt} [SEP] {response}"
        texts.append(combined_text)
    
    # Generate embeddings
    print(f"Encoding {len(texts)} text samples...")
    text_embeddings = encoder.encode(
        texts, 
        show_progress_bar=True,
        batch_size=32, 
        convert_to_numpy=True
    )
    
    # Cache embeddings if path provided
    if cache_file_path:
        np.save(cache_file_path, text_embeddings)
    
    return text_embeddings


def prepare_features(data, metric_to_embedding, text_embeddings):
    """
    Prepare metric and text features for model training.
    
    Args:
        data: List of samples
        metric_to_embedding: Dictionary mapping metric names to embeddings
        text_embeddings: Pre-computed text embeddings
        
    Returns:
        tuple: (metric_features, text_features)
    """
    metric_features = np.array([
        metric_to_embedding[sample['metric_name']] for sample in data
    ])
    return metric_features, text_embeddings


# ============================================================================
# SIGNAL 3: NLI-BASED COHERENCE SCORER
# ============================================================================

class NLICoherenceScorer:
    """
    Uses Natural Language Inference (NLI) to measure prompt-response coherence.
    
    NLI models output logits for three classes: [contradiction, neutral, entailment]
    We use the entailment probability as the coherence score, which measures how
    well the response logically follows from the prompt.
    
    Attributes:
        tokenizer: Hugging Face tokenizer for the NLI model
        model: Pre-trained NLI model
        device: Computing device (cuda/cpu)
    """

    def __init__(self, model_name="MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli", device=None):
        """
        Initialize NLI coherence scorer.
        
        Args:
            model_name: Name of pre-trained NLI model from Hugging Face
            device: Computing device ('cuda' or 'cpu'). Auto-detects if None.
        """
        print(f"\n  Loading NLI Model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode for inference
        self.model_name = model_name
        print(f"  ✓ NLI Model loaded successfully on {self.device}")

    def predict_coherence(self, prompts, responses, batch_size=32, cache_path=None):
        """
        Compute coherence scores using NLI entailment probability.
        
        Args:
            prompts: List of premise texts (prompts)
            responses: List of hypothesis texts (responses)
            batch_size: Batch size for inference
            cache_path: Optional path to cache computed scores
        
        Returns:
            np.ndarray: Array of coherence scores (entailment probabilities)
        """
        # Load from cache if available
        if cache_path and os.path.exists(cache_path):
            print(f"  Loading cached NLI coherence scores from {cache_path}")
            return np.load(cache_path)

        # Clean and prepare inputs
        prompts_clean = [str(p) if p is not None else "" for p in prompts]
        responses_clean = [str(r) if r is not None else "" for r in responses]
        n = len(prompts_clean)
        coherence_scores = []

        print(f"  Computing NLI coherence for {n} pairs...")

        # Process in batches
        for i in range(0, n, batch_size):
            batch_prompts = prompts_clean[i:i + batch_size]
            batch_responses = responses_clean[i:i + batch_size]

            # Tokenize: premise (prompt) and hypothesis (response)
            inputs = self.tokenizer(
                batch_prompts,
                batch_responses,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            ).to(self.device)

            # Forward pass (no gradient computation for inference)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # Shape: (batch_size, 3)

            # Apply softmax to convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Extract entailment probability (typically index 2)
            entailment_probs = probs[:, 2].cpu().numpy()
            coherence_scores.extend(entailment_probs)

            # Progress update every 10 batches
            if (i // batch_size + 1) % 10 == 0:
                print(f"    Processed {min(i + batch_size, n)}/{n} pairs...")

        coherence_scores = np.array(coherence_scores)

        # Cache results if path provided
        if cache_path:
            np.save(cache_path, coherence_scores)

        print(f"  ✓ NLI Coherence scores: min={coherence_scores.min():.3f}, "
              f"max={coherence_scores.max():.3f}, mean={coherence_scores.mean():.3f}")
        
        return coherence_scores


# ============================================================================
# SIGNAL 2: METRIC-MATCHER CLASSIFIER
# ============================================================================

class MetricMatcherClassifier:
    """
    Neural network classifier to predict if a metric-text pair is a true match.
    
    This is more sophisticated than simple cosine similarity as it learns
    complex interaction patterns between metric and text embeddings.
    
    Attributes:
        metric_dim: Dimensionality of metric embeddings
        text_dim: Dimensionality of text embeddings
        hidden_dims: List of hidden layer dimensions
        model: Keras model instance
        scaler: MinMaxScaler for normalization
    """
    
    def __init__(self, metric_dim=768, text_dim=768, hidden_dims=[512, 256, 128]):
        """
        Initialize the metric-matcher classifier.
        
        Args:
            metric_dim: Dimension of metric embeddings
            text_dim: Dimension of text embeddings
            hidden_dims: List of hidden layer sizes
        """
        self.metric_dim = metric_dim
        self.text_dim = text_dim
        self.hidden_dims = hidden_dims
        self.model = None
        self.scaler = MinMaxScaler()
    
    def build_model(self):
        """
        Build the neural network architecture for metric-text matching.
        
        Architecture:
            - Two input branches for metric and text embeddings
            - Interaction features (multiplication and difference)
            - Deep fully-connected layers with dropout
            - Sigmoid output for binary classification
            
        Returns:
            keras.Model: Compiled model ready for training
        """
        # Input layers
        metric_input = layers.Input(shape=(self.metric_dim,), name='metric_input')
        text_input = layers.Input(shape=(self.text_dim,), name='text_input')
        
        # Concatenate embeddings
        concat = layers.Concatenate()([metric_input, text_input])
        
        # Create interaction features
        interaction = layers.Multiply()([metric_input, text_input])
        difference = layers.Subtract()([metric_input, text_input])
        
        # Combine all features
        combined = layers.Concatenate()([concat, interaction, difference])
        
        # Deep network with batch normalization and dropout
        x = combined
        for hidden_dim in self.hidden_dims:
            x = layers.Dense(hidden_dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
        
        # Output layer - probability of match
        output = layers.Dense(1, activation='sigmoid', name='match_probability')(x)
        
        # Create and compile model
        model = Model(inputs=[metric_input, text_input], outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        self.model = model
        return model
    
    def create_training_data(self, X_metric, X_text, negative_ratio=2.0):
        """
        Create positive and negative pairs for binary classification training.
        
        Args:
            X_metric: Metric embeddings [n_samples, metric_dim]
            X_text: Text embeddings [n_samples, text_dim]
            negative_ratio: Ratio of negative to positive samples
        
        Returns:
            tuple: (X_metric_pairs, X_text_pairs, labels)
        """
        n_samples = len(X_metric)
        n_negatives = int(n_samples * negative_ratio)
        
        print(f"\n  Creating training data for Metric-Matcher:")
        print(f"    Positive pairs: {n_samples}")
        print(f"    Negative pairs: {n_negatives}")
        
        # Positive pairs (actual matches from dataset)
        X_metric_pos = X_metric.copy()
        X_text_pos = X_text.copy()
        labels_pos = np.ones(n_samples)
        
        # Negative pairs (random mismatches)
        X_metric_neg = []
        X_text_neg = []
        
        for i in range(n_negatives):
            idx1 = np.random.randint(0, n_samples)
            idx2 = np.random.randint(0, n_samples)
            
            # Ensure indices are different
            while idx1 == idx2:
                idx2 = np.random.randint(0, n_samples)
            
            X_metric_neg.append(X_metric[idx1])
            X_text_neg.append(X_text[idx2])
        
        X_metric_neg = np.array(X_metric_neg)
        X_text_neg = np.array(X_text_neg)
        labels_neg = np.zeros(n_negatives)
        
        # Combine positive and negative pairs
        X_metric_all = np.vstack([X_metric_pos, X_metric_neg])
        X_text_all = np.vstack([X_text_pos, X_text_neg])
        labels_all = np.concatenate([labels_pos, labels_neg])
        
        # Shuffle the combined dataset
        indices = np.arange(len(labels_all))
        np.random.shuffle(indices)
        
        X_metric_all = X_metric_all[indices]
        X_text_all = X_text_all[indices]
        labels_all = labels_all[indices]
        
        print(f"    Total pairs: {len(labels_all)}")
        print(f"    Class balance: {labels_all.mean():.3f}")
        
        return X_metric_all, X_text_all, labels_all
    
    def train(self, X_metric, X_text, epochs=20, batch_size=64, validation_split=0.15):
        """
        Train the metric-matcher classifier.
        
        Args:
            X_metric: Metric embeddings
            X_text: Text embeddings
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            
        Returns:
            keras.History: Training history object
        """
        print("\n" + "=" * 70)
        print("TRAINING METRIC-MATCHER CLASSIFIER (S2)")
        print("=" * 70)
        
        # Create training data with positive and negative pairs
        X_metric_pairs, X_text_pairs, labels = self.create_training_data(
            X_metric, X_text, negative_ratio=2.0
        )
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        print("\n  Model architecture:")
        self.model.summary()
        
        # Define callbacks for training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=3, 
                min_lr=1e-6
            )
        ]
        
        # Train the model
        print("\n  Training...")
        history = self.model.fit(
            {'metric_input': X_metric_pairs, 'text_input': X_text_pairs},
            labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("  ✓ Metric-Matcher trained successfully!")
        
        return history
    
    def predict_relevance(self, X_metric, X_text, batch_size=256):
        """
        Predict relevance scores for metric-text pairs.
        
        Args:
            X_metric: Metric embeddings
            X_text: Text embeddings
            batch_size: Batch size for prediction
        
        Returns:
            np.ndarray: Relevance scores in range [0, 1]
        """
        print(f"  Predicting relevance for {len(X_metric)} pairs...")
        
        relevance_scores = self.model.predict(
            {'metric_input': X_metric, 'text_input': X_text},
            batch_size=batch_size,
            verbose=0
        ).flatten()
        
        print(f"  ✓ Relevance scores: min={relevance_scores.min():.3f}, "
              f"max={relevance_scores.max():.3f}, mean={relevance_scores.mean():.3f}")
        
        return relevance_scores


# ============================================================================
# SIGNAL 1: AUTOENCODER FOR ANOMALY DETECTION
# ============================================================================

def build_smart_autoencoder(metric_dim=768, text_dim=768, bottleneck_dim=48):
    """
    Build an autoencoder for detecting anomalous metric-text pairs.
    
    The autoencoder learns to reconstruct normal (high-quality) samples.
    Higher reconstruction errors indicate anomalies or low-quality samples.
    
    Args:
        metric_dim: Dimension of metric embeddings
        text_dim: Dimension of text embeddings
        bottleneck_dim: Dimension of the bottleneck layer
        
    Returns:
        keras.Model: Compiled autoencoder model
    """
    # Input layers
    metric_input = layers.Input(shape=(metric_dim,), name='metric_input')
    text_input = layers.Input(shape=(text_dim,), name='text_input')
    
    # Create interaction features
    interaction_product = layers.Multiply()([metric_input, text_input])
    interaction_diff = layers.Subtract()([metric_input, text_input])
    
    # Fuse all features
    fused_input = layers.Concatenate()([
        metric_input, text_input, interaction_product, interaction_diff
    ])
    
    # Encoder path
    x = layers.Dense(1024, activation='relu')(fused_input)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    bottleneck = layers.Dense(bottleneck_dim, activation='relu', name='bottleneck')(x)
    
    # Decoder path
    x = layers.Dense(512, activation='relu')(bottleneck)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1024, activation='relu')(x)
    
    # Reconstruction output
    original_dims = fused_input.shape[1]
    reconstruction = layers.Dense(original_dims, activation='linear', name='reconstruction')(x)
    
    # Create and compile model
    autoencoder = Model(inputs=[metric_input, text_input], outputs=reconstruction)
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001), 
        loss='huber'
    )
    
    return autoencoder


def train_autoencoder_smart(model, X_metric_all, X_text_all, all_scores,
                            score_threshold=8.0, epochs=50, batch_size=32):
    """
    Train autoencoder on high-quality samples only (semi-supervised approach).
    
    By training only on normal samples (score >= threshold), the autoencoder
    learns to reconstruct high-quality pairs. Low-quality pairs will have
    higher reconstruction errors.
    
    Args:
        model: Autoencoder model to train
        X_metric_all: All metric embeddings
        X_text_all: All text embeddings
        all_scores: Quality scores for all samples
        score_threshold: Minimum score to consider a sample as "normal"
        epochs: Maximum training epochs
        batch_size: Training batch size
        
    Returns:
        tuple: (trained_model, training_history)
    """
    # Filter for high-quality samples
    normal_indices = np.where(all_scores >= score_threshold)[0]
    print(f"Original training samples: {len(all_scores)}")
    print(f"Filtered for 'normal' data (score >= {score_threshold}): {len(normal_indices)} samples")
    
    X_metric_normal = X_metric_all[normal_indices]
    X_text_normal = X_text_all[normal_indices]
    
    # Create target features (reconstruction targets)
    interaction_product_normal = X_metric_normal * X_text_normal
    interaction_diff_normal = X_metric_normal - X_text_normal
    
    Y_target_normal = np.concatenate([
        X_metric_normal, X_text_normal, 
        interaction_product_normal, interaction_diff_normal
    ], axis=1)
    
    # Split into train/validation
    split_idx = int(len(X_metric_normal) * 0.9)
    X_train_inputs = {
        'metric_input': X_metric_normal[:split_idx], 
        'text_input': X_text_normal[:split_idx]
    }
    Y_train_target = Y_target_normal[:split_idx]
    X_val_inputs = {
        'metric_input': X_metric_normal[split_idx:], 
        'text_input': X_text_normal[split_idx:]
    }
    Y_val_target = Y_target_normal[split_idx:]
    
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6
        )
    ]
    
    # Train the autoencoder
    history = model.fit(
        X_train_inputs, Y_train_target,
        validation_data=(X_val_inputs, Y_val_target),
        epochs=epochs, 
        batch_size=batch_size,
        callbacks=callbacks, 
        verbose=1, 
        shuffle=True
    )
    
    print("  ✓ Autoencoder (Anomaly Expert) trained!")
    return model, history


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def augment_with_synthetic_negatives(X_metric, X_text, prompts, responses, scores, 
                                     total_augment_ratio=0.75, partial_failure_pct=0.3):
    """
    Create synthetic low-quality examples using a hybrid failure strategy.
    
    This augmentation creates two types of synthetic failures:
    1. Partial Failures (scores 4-8): Either S2 or S3 fails, but not both
    2. Total Failures (scores 0-3): Both S2 and S3 fail
    
    Args:
        X_metric: Metric embeddings
        X_text: Text embeddings
        prompts: Prompt texts
        responses: Response texts
        scores: Quality scores
        total_augment_ratio: Fraction of synthetic samples to create
        partial_failure_pct: Percentage of synthetic samples that are partial failures
        
    Returns:
        tuple: Augmented (X_metric, X_text, prompts, responses, scores)
    """
    n_samples = len(scores)
    n_synthetic_total = int(n_samples * total_augment_ratio)
    n_synthetic_partial = int(n_synthetic_total * partial_failure_pct)
    n_synthetic_total_fail = n_synthetic_total - n_synthetic_partial
    
    print(f"\n--- Augmenting with Hybrid Strategy ---")
    print(f"  Total new samples: {n_synthetic_total}")
    print(f"    Partial Failures (4.0-8.0): {n_synthetic_partial}")
    print(f"    Total Failures (0.0-3.0):   {n_synthetic_total_fail}")

    # Create shuffled indices for mismatching
    indices = np.arange(n_samples)
    shuffled_indices = np.random.permutation(indices)
    
    # Ensure no index matches itself
    for i in range(n_samples):
        if indices[i] == shuffled_indices[i]:
            swap_idx = (i + 1) % n_samples
            shuffled_indices[i], shuffled_indices[swap_idx] = shuffled_indices[swap_idx], shuffled_indices[i]
    
    syn_metric, syn_text, syn_prompts, syn_responses, syn_scores = [], [], [], [], []

    # --- Create Partial Failures (S2-fail OR S3-fail) ---
    idx1 = np.random.choice(indices, n_synthetic_partial, replace=True)
    idx2 = np.random.choice(shuffled_indices, n_synthetic_partial, replace=True)
    
    for i in range(n_synthetic_partial):
        if np.random.rand() > 0.5:
            # S2 Failure (Relevance mismatch)
            syn_metric.append(X_metric[idx1[i]])
            syn_text.append(X_text[idx2[i]])  # Mismatched text
            syn_prompts.append(prompts[idx1[i]])
            syn_responses.append(responses[idx1[i]])
            # Score follows normal distribution around 7.0
            score = np.random.normal(loc=7.0, scale=1.0)
            syn_scores.append(np.clip(score, 4.0, 9.5))
        else:
            # S3 Failure (Coherence mismatch)
            syn_metric.append(X_metric[idx1[i]])
            syn_text.append(X_text[idx1[i]])
            syn_prompts.append(prompts[idx1[i]])
            syn_responses.append(responses[idx2[i]])  # Mismatched response
            # Score follows normal distribution around 4.0
            score = np.random.normal(loc=4.0, scale=1.2)
            syn_scores.append(np.clip(score, 1.0, 7.0))

    # --- Create Total Failures (S2-fail AND S3-fail) ---
    idx1 = np.random.choice(indices, n_synthetic_total_fail, replace=True)
    idx2 = np.random.choice(shuffled_indices, n_synthetic_total_fail, replace=True)

    for i in range(n_synthetic_total_fail):
        syn_metric.append(X_metric[idx1[i]])
        syn_text.append(X_text[idx2[i]])  # Mismatched text
        syn_prompts.append(prompts[idx1[i]])
        syn_responses.append(responses[idx2[i]])  # Mismatched response
        # Score follows normal distribution around 1.0
        score = np.random.normal(loc=1.0, scale=1.0)
        syn_scores.append(np.clip(score, 0.0, 3.5))

    print(f"  ✓ Augmentation complete.")

    # Combine original and synthetic data
    X_metric_aug = np.vstack([X_metric, np.array(syn_metric)])
    X_text_aug = np.vstack([X_text, np.array(syn_text)])
    prompts_aug = np.concatenate([prompts, np.array(syn_prompts)])
    responses_aug = np.concatenate([responses, np.array(syn_responses)])
    scores_aug = np.concatenate([scores, np.array(syn_scores)])
    
    # Shuffle the augmented dataset
    print("  Shuffling augmented dataset...")
    X_metric_aug, X_text_aug, prompts_aug, responses_aug, scores_aug = shuffle(
        X_metric_aug, X_text_aug, prompts_aug, responses_aug, scores_aug, random_state=42
    )
    
    print(f"  ✓ Augmented dataset: {n_samples} → {len(scores_aug)} samples")
    return X_metric_aug, X_text_aug, prompts_aug, responses_aug, scores_aug


def get_anomaly_errors(autoencoder, X_metric, X_text):
    """
    Compute reconstruction errors from autoencoder.
    
    Higher errors indicate anomalous or low-quality samples.
    
    Args:
        autoencoder: Trained autoencoder model
        X_metric: Metric embeddings
        X_text: Text embeddings
        
    Returns:
        np.ndarray: Reconstruction errors (MSE per sample)
    """
    ae_input_list = [X_metric, X_text]
    reconstructed = autoencoder.predict(ae_input_list, verbose=0, batch_size=256)
    
    # Recreate original fused features
    product = X_metric * X_text
    diff = X_metric - X_text
    original_fused = np.concatenate([X_metric, X_text, product, diff], axis=1)
    
    # Compute mean squared error per sample
    errors = np.mean(np.power(original_fused - reconstructed, 2), axis=1)
    return errors.reshape(-1, 1)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_enhanced_features(df_base):
    """
    Create interaction and statistical features from base signals.
    
    Args:
        df_base: DataFrame with S1_Anomaly, S2_Relevance, S3_Coherence columns
        
    Returns:
        pd.DataFrame: Enhanced feature set
    """
    df = df_base.copy()
    
    # Interaction features (pairwise and triple)
    df['S1_x_S2'] = df['S1_Anomaly'] * df['S2_Relevance']
    df['S1_x_S3'] = df['S1_Anomaly'] * df['S3_Coherence']
    df['S2_x_S3'] = df['S2_Relevance'] * df['S3_Coherence']
    df['S1_x_S2_x_S3'] = df['S1_Anomaly'] * df['S2_Relevance'] * df['S3_Coherence']
    
    # Polynomial features
    df['S1_squared'] = df['S1_Anomaly'] ** 2
    df['S2_squared'] = df['S2_Relevance'] ** 2
    df['S3_squared'] = df['S3_Coherence'] ** 2
    
    # Statistical aggregation features
    df['mean_signals'] = df[['S1_Anomaly', 'S2_Relevance', 'S3_Coherence']].mean(axis=1)
    df['std_signals'] = df[['S1_Anomaly', 'S2_Relevance', 'S3_Coherence']].std(axis=1)
    df['min_signals'] = df[['S1_Anomaly', 'S2_Relevance', 'S3_Coherence']].min(axis=1)
    df['max_signals'] = df[['S1_Anomaly', 'S2_Relevance', 'S3_Coherence']].max(axis=1)
    
    return df


# ============================================================================
# ENSEMBLE META-MODELS
# ============================================================================

class EnsembleMetaModels:
    """
    Stacked ensemble of gradient boosting models with Ridge meta-learner.
    
    Uses K-fold out-of-fold predictions to train a Ridge regression model
    that combines predictions from multiple base models.
    
    Attributes:
        use_enhanced_features: Whether to use engineered features
        n_folds: Number of folds for out-of-fold prediction generation
        models: Dictionary of base models
        stacking_model: Ridge regression meta-learner
        oof_predictions: Out-of-fold predictions from base models
    """
    
    def __init__(self, use_enhanced_features=True, n_folds=5):
        """
        Initialize ensemble.
        
        Args:
            use_enhanced_features: Whether to create interaction/polynomial features
            n_folds: Number of K-fold splits for OOF predictions
        """
        self.use_enhanced_features = use_enhanced_features
        self.n_folds = n_folds
        self.models = {}
        self.stacking_model = None
        
    def _initialize_models(self):
        """
        Initialize base models with regularized hyperparameters.
        
        Models are tuned to prevent overfitting through:
        - Limited depth
        - Regularization parameters
        - Subsampling
        """
        self.models = {
            'xgb_1': xgb.XGBRegressor(
                objective='reg:squarederror', 
                max_depth=3,           # Shallow trees to prevent overfitting
                learning_rate=0.03,
                n_estimators=2000, 
                subsample=0.7,         # Sample 70% of data per tree
                colsample_bytree=0.7,  # Sample 70% of features per tree
                reg_alpha=1.0,         # L1 regularization
                reg_lambda=2.0,        # L2 regularization
                random_state=42,
                tree_method='hist', 
                early_stopping_rounds=100
            ),
            'lgb_1': lgb.LGBMRegressor(
                objective='regression', 
                num_leaves=31,
                learning_rate=0.03,
                n_estimators=2000, 
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42, 
                verbose=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=300,
                max_depth=15,          # Limit tree depth
                min_samples_split=5,   # Minimum samples to split
                random_state=42, 
                n_jobs=-1
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.05, 
                max_depth=5,
                subsample=0.8,
                random_state=42
            ),
        }
    
    def fit(self, X, y, verbose=True):
        """
        Train the stacked ensemble using out-of-fold predictions.
        
        Process:
        1. Generate out-of-fold predictions from base models using K-fold CV
        2. Train Ridge meta-learner on OOF predictions
        3. Retrain base models on full dataset
        
        Args:
            X: Feature DataFrame
            y: Target scores
            verbose: Whether to print training progress
            
        Returns:
            self: Fitted ensemble
        """
        print("\n" + "=" * 70)
        print("TRAINING STACKED META-MODEL ENSEMBLE")
        print("=" * 70)
        
        self._initialize_models()
        
        # Apply feature engineering if enabled
        if self.use_enhanced_features:
            X = create_enhanced_features(X)
        
        # STEP 1: Generate Out-of-Fold predictions
        print("\n[STEP 1/3] Generating Out-of-Fold predictions from base models...")
        self.oof_predictions = {name: np.zeros(len(y)) for name in self.models.keys()}
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"  Processing fold {fold_idx + 1}/{self.n_folds}...")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            for model_name, model in self.models.items():
                # Clone and train model for this fold
                if model_name.startswith('xgb'):
                    fold_model = xgb.XGBRegressor(**model.get_params())
                    fold_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                elif model_name.startswith('lgb'):
                    fold_model = lgb.LGBMRegressor(**model.get_params())
                    fold_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                                 callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
                else:
                    fold_model = type(model)(**model.get_params())
                    fold_model.fit(X_train, y_train)
                
                # Store OOF predictions
                self.oof_predictions[model_name][val_idx] = fold_model.predict(X_val)
        
        # Calculate individual model performance
        print("\n  Base model OOF performance:")
        oof_rmse_scores = {}
        for model_name in self.models.keys():
            oof_rmse = np.sqrt(mean_squared_error(y, self.oof_predictions[model_name]))
            oof_rmse_scores[model_name] = oof_rmse
            print(f"    {model_name}: RMSE = {oof_rmse:.4f}")
        
        # STEP 2: Train Ridge stacking model
        print("\n[STEP 2/3] Training Ridge stacking model on OOF predictions...")
        
        # Create DataFrame from OOF predictions
        oof_df = pd.DataFrame(self.oof_predictions)
        
        # Train RidgeCV with cross-validation to find optimal alpha
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        self.stacking_model = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
        self.stacking_model.fit(oof_df, y)
        
        # Evaluate stacking model
        stacked_oof_pred = self.stacking_model.predict(oof_df)
        stacked_oof_rmse = np.sqrt(mean_squared_error(y, stacked_oof_pred))
        
        print(f"  ✓ Ridge stacking model trained!")
        print(f"    Optimal alpha: {self.stacking_model.alpha_:.4f}")
        print(f"    Stacked OOF RMSE: {stacked_oof_rmse:.4f}")
        print(f"    Best base model RMSE: {min(oof_rmse_scores.values()):.4f}")
        print(f"    Improvement: {min(oof_rmse_scores.values()) - stacked_oof_rmse:.4f}")
        
        # Print Ridge coefficients (model weights)
        print("\n  Ridge model weights for base models:")
        for model_name, coef in zip(self.models.keys(), self.stacking_model.coef_):
            print(f"    {model_name}: {coef:.4f}")
        print(f"    Intercept: {self.stacking_model.intercept_:.4f}")
        
        # STEP 3: Retrain base models on full data
        print("\n[STEP 3/3] Retraining base models on full dataset...")
        for model_name, model in self.models.items():
            # Models with early stopping need validation set
            if model_name.startswith('xgb') or model_name.startswith('lgb'):
                X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.1, random_state=42)
                if model_name.startswith('xgb'):
                    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
                else:
                    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                             callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            else:
                model.fit(X, y)
        
        print("  ✓ All base models retrained on full data!")
        print("\n" + "=" * 70)
        print("STACKED ENSEMBLE TRAINING COMPLETE")
        print("=" * 70)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the stacked ensemble.
        
        Process:
        1. Get predictions from all base models
        2. Combine using Ridge stacking model
        
        Args:
            X: Feature DataFrame
            
        Returns:
            np.ndarray: Final predictions
        """
        # Apply feature engineering if enabled
        if self.use_enhanced_features:
            X = create_enhanced_features(X)
        
        # Get predictions from all base models
        base_predictions = {}
        for model_name, model in self.models.items():
            base_predictions[model_name] = model.predict(X)
        
        # Create DataFrame from base predictions
        base_pred_df = pd.DataFrame(base_predictions)
        
        # Use stacking model for final prediction
        stacked_pred = self.stacking_model.predict(base_pred_df)
        
        return stacked_pred


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main_enhanced_workflow():
    """
    Main training and prediction pipeline.
    
    Steps:
    1. Load data and embeddings
    2. Encode text using SentenceTransformers
    3. Train autoencoder (S1) on high-quality samples
    4. Train metric-matcher classifier (S2)
    5. Compute NLI-based coherence scores (S3)
    6. Augment training data with synthetic negatives
    7. Train ensemble meta-models
    8. Generate predictions
    9. Save outputs and visualizations
    
    Returns:
        tuple: (ensemble, autoencoder, metric_matcher, nli_scorer, analysis_df)
    """
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    
    metric_to_embedding, train_data, test_data = load_data()
    y_train_scores = np.array([sample['score'] for sample in train_data], dtype=np.float32)
    
    # ========================================================================
    # STEP 2: ENCODE TEXT DATA
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: ENCODING TEXT DATA")
    print("=" * 70)
    
    train_conv_emb = encode_text_data(
        train_data, 
        cache_file_path=r'C:\Users\Siddharth Nair\Downloads\train_embeddings.npy'
    )
    test_conv_emb = encode_text_data(
        test_data,
        cache_file_path=r'C:\Users\Siddharth Nair\Downloads\test_embeddings.npy'
    )
    
    X_metric_train, X_text_train = prepare_features(train_data, metric_to_embedding, train_conv_emb)
    X_metric_test, X_text_test = prepare_features(test_data, metric_to_embedding, test_conv_emb)
    
    # Extract prompts and responses for coherence scoring
    train_prompts = np.array([str(s.get('prompt', '')) for s in train_data])
    train_responses = np.array([str(s.get('expected_response', s.get('response', ''))) for s in train_data])
    test_prompts = np.array([str(s.get('prompt', '')) for s in test_data])
    test_responses = np.array([str(s.get('response', '')) for s in test_data])
    
    # ========================================================================
    # STEP 3: TRAIN AUTOENCODER (S1)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING AUTOENCODER (S1)")
    print("=" * 70)
    
    autoencoder = build_smart_autoencoder()
    autoencoder, _ = train_autoencoder_smart(
        autoencoder, X_metric_train, X_text_train, y_train_scores
    )
    
    # ========================================================================
    # STEP 4: TRAIN METRIC-MATCHER (S2)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING METRIC-MATCHER (S2)")
    print("=" * 70)
    
    metric_matcher = MetricMatcherClassifier()
    metric_matcher.train(X_metric_train, X_text_train)
    
    # ========================================================================
    # STEP 5: COMPUTE NLI-BASED COHERENCE (S3)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: COMPUTING NLI-BASED COHERENCE (S3)")
    print("=" * 70)
    
    nli_scorer = NLICoherenceScorer("MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli")
    
    # Calculate S3 for train data (cached for efficiency)
    S3_train_coherence_cached = nli_scorer.predict_coherence(
        train_prompts, train_responses,
        cache_path=r'C:\Users\Siddharth Nair\Downloads\train_coherence_nli.npy'
    )
    
    # Calculate S3 for test data
    S3_test_coherence = nli_scorer.predict_coherence(
        test_prompts, test_responses,
        cache_path=r'C:\Users\Siddharth Nair\Downloads\test_coherence_nli.npy'
    )
    
    # ========================================================================
    # STEP 6: AUGMENTATION & FEATURE GENERATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: AUGMENTING & GENERATING META-FEATURES")
    print("=" * 70)
    
    # Augment training data with synthetic negatives
    X_metric_aug, X_text_aug, prompts_aug, responses_aug, y_scores_aug = \
        augment_with_synthetic_negatives(
            X_metric_train, X_text_train, 
            train_prompts, train_responses, 
            y_train_scores, 
            total_augment_ratio=0.5, 
            partial_failure_pct=0.25
        )
    
    # Fit S1 scaler on original data only (to prevent data leakage)
    print("\n  Fitting S1 Scaler on ORIGINAL data only...")
    S1_train_errors_original = get_anomaly_errors(autoencoder, X_metric_train, X_text_train)
    scaler_anom = MinMaxScaler()
    scaler_anom.fit(S1_train_errors_original)
    
    # Generate S1, S2, S3 for augmented training data
    print("\n  Generating signals for augmented training data...")
    S1_train_errors_aug = get_anomaly_errors(autoencoder, X_metric_aug, X_text_aug)
    S1_train_anomaly = 1.0 - scaler_anom.transform(S1_train_errors_aug).flatten()
    
    S2_train_relevance = metric_matcher.predict_relevance(X_metric_aug, X_text_aug)
    
    # Calculate S3 for all augmented data
    print("  Calculating S3 NLI Coherence for ALL augmented data...")
    S3_train_coherence_aug = nli_scorer.predict_coherence(
        prompts_aug, responses_aug,
        batch_size=64,
        cache_path=None
    )
    
    # Create meta-feature DataFrame for training
    X_meta_train = pd.DataFrame({
        'S1_Anomaly': S1_train_anomaly,
        'S2_Relevance': S2_train_relevance,
        'S3_Coherence': S3_train_coherence_aug
    })
    
    # Generate S1, S2, S3 for test data
    print("\n  Generating signals for test data...")
    S1_test_errors = get_anomaly_errors(autoencoder, X_metric_test, X_text_test)
    S1_test_anomaly = 1.0 - scaler_anom.transform(S1_test_errors).flatten()
    
    S2_test_relevance = metric_matcher.predict_relevance(X_metric_test, X_text_test)
    
    # Create meta-feature DataFrame for testing
    X_meta_test = pd.DataFrame({
        'S1_Anomaly': S1_test_anomaly,
        'S2_Relevance': S2_test_relevance,
        'S3_Coherence': S3_test_coherence
    })
    
    print(f"\n  Meta-features summary:")
    print(f"    Train shape: {X_meta_train.shape}")
    print(f"    Test shape: {X_meta_test.shape}")
    print(f"\n    Train statistics:")
    print(X_meta_train.describe())
    print(f"\n    Test statistics:")
    print(X_meta_test.describe())
    
    # ========================================================================
    # STEP 7: TRAIN ENSEMBLE META-MODELS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: TRAINING ENSEMBLE META-MODELS")
    print("=" * 70)
    
    ensemble = EnsembleMetaModels(use_enhanced_features=True, n_folds=5)
    ensemble.fit(X_meta_train, y_scores_aug, verbose=True)
    
    # ========================================================================
    # STEP 8: GENERATE PREDICTIONS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: GENERATING PREDICTIONS")
    print("=" * 70)
    
    final_predictions = ensemble.predict(X_meta_test)
    final_predictions_clipped = np.clip(final_predictions, 0, 10)
    
    print(f"\nFinal predictions:")
    print(f"  Range: [{final_predictions_clipped.min():.2f}, {final_predictions_clipped.max():.2f}]")
    print(f"  Mean: {final_predictions_clipped.mean():.2f}")
    print(f"  Std: {final_predictions_clipped.std():.2f}")
    
    # ========================================================================
    # SIGNAL ANALYSIS
    # ========================================================================
    print("\n" + "=" * 70)
    print("SIGNAL ANALYSIS")
    print("=" * 70)
    
    print("\nTest Set Signal Statistics:")
    print(f"  S1 (Anomaly):       mean={S1_test_anomaly.mean():.3f}, std={S1_test_anomaly.std():.3f}")
    print(f"  S2 (Relevance):     mean={S2_test_relevance.mean():.3f}, std={S2_test_relevance.std():.3f}")
    print(f"  S3 (NLI Coherence): mean={S3_test_coherence.mean():.3f}, std={S3_test_coherence.std():.3f}")
    
    print("\nTrain Set Signal Statistics (original data only):")
    n_orig = len(train_data)
    print(f"  S1 (Anomaly):       mean={S1_train_anomaly[:n_orig].mean():.3f}, std={S1_train_anomaly[:n_orig].std():.3f}")
    print(f"  S2 (Relevance):     mean={S2_train_relevance[:n_orig].mean():.3f}, std={S2_train_relevance[:n_orig].std():.3f}")
    print(f"  S3 (NLI Coherence): mean={S3_train_coherence_aug[:n_orig].mean():.3f}, std={S3_train_coherence_aug[:n_orig].std():.3f}")
    
    # Identify potentially problematic samples
    low_quality_mask = (S1_test_anomaly < 0.3) | (S2_test_relevance < 0.3) | (S3_test_coherence < 0.3)
    print(f"\nPotentially low-quality samples (any signal < 0.3): {low_quality_mask.sum()} / {len(S1_test_anomaly)}")
    
    # ========================================================================
    # STEP 9: SAVE OUTPUTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 9: SAVING OUTPUTS")
    print("=" * 70)
    
    submission_path = r'C:\Users\Siddharth Nair\Downloads\sample_submission.csv'
    output_path = r'C:\Users\Siddharth Nair\Downloads\submission_enhanced_model_v3_nli_normal_0.75.csv'
    
    submission = pd.read_csv(submission_path)
    submission['score'] = final_predictions_clipped
    submission.to_csv(output_path, index=False)
    print(f"✓ Submission saved to {output_path}")
    
    # Save detailed analysis
    analysis_df = pd.DataFrame({
        'S1_Anomaly': S1_test_anomaly,
        'S2_Relevance': S2_test_relevance,
        'S3_NLI_Coherence': S3_test_coherence,
        'prediction': final_predictions_clipped,
        'low_quality_flag': low_quality_mask
    })
    analysis_path = r'C:\Users\Siddharth Nair\Downloads\test_signal_analysis_v3_nli_0.75.csv'
    analysis_df.to_csv(analysis_path, index=False)
    print(f"✓ Signal analysis saved to {analysis_path}")
    
    # ========================================================================
    # STEP 10: CREATE VISUALIZATIONS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 10: CREATING VISUALIZATIONS")
    print("=" * 70)
    
    try:
        # Create 2x3 subplot for signal distributions and correlations
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Row 1: Signal distributions
        axes[0, 0].hist(S1_test_anomaly, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[0, 0].set_title('S1: Anomaly Score Distribution')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(S1_test_anomaly.mean(), color='black', linestyle='--', label='Mean')
        axes[0, 0].legend()
        
        axes[0, 1].hist(S2_test_relevance, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].set_title('S2: Metric-Text Relevance (Trained Matcher)')
        axes[0, 1].set_xlabel('Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(S2_test_relevance.mean(), color='black', linestyle='--', label='Mean')
        axes[0, 1].legend()
        
        axes[0, 2].hist(S3_test_coherence, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 2].set_title('S3: Prompt-Response Coherence (NLI Entailment)')
        axes[0, 2].set_xlabel('Score')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(S3_test_coherence.mean(), color='black', linestyle='--', label='Mean')
        axes[0, 2].legend()
        
        # Row 2: Signal correlations with predictions
        axes[1, 0].scatter(S1_test_anomaly, final_predictions_clipped, alpha=0.3, s=10, c='red')
        axes[1, 0].set_xlabel('S1: Anomaly Score')
        axes[1, 0].set_ylabel('Predicted Score')
        axes[1, 0].set_title('S1 vs Prediction')
        
        axes[1, 1].scatter(S2_test_relevance, final_predictions_clipped, alpha=0.3, s=10, c='blue')
        axes[1, 1].set_xlabel('S2: Relevance Score')
        axes[1, 1].set_ylabel('Predicted Score')
        axes[1, 1].set_title('S2 vs Prediction')
        
        axes[1, 2].scatter(S3_test_coherence, final_predictions_clipped, alpha=0.3, s=10, c='green')
        axes[1, 2].set_xlabel('S3: NLI Coherence Score')
        axes[1, 2].set_ylabel('Predicted Score')
        axes[1, 2].set_title('S3 vs Prediction')
        
        plt.tight_layout()
        plot_path = r'C:\Users\Siddharth Nair\Downloads\enhanced_model_analysis_v3_nli_normal_0.75.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Analysis plots saved to {plot_path}")
        plt.close()
        
        # 3D scatter plot of signal space
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            S1_test_anomaly, S2_test_relevance, S3_test_coherence,
            c=final_predictions_clipped, cmap='viridis', s=20, alpha=0.6
        )
        
        ax.set_xlabel('S1: Anomaly')
        ax.set_ylabel('S2: Relevance')
        ax.set_zlabel('S3: NLI Coherence')
        ax.set_title('Signal Space colored by Predicted Score')
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Predicted Score', rotation=270, labelpad=20)
        
        plot_3d_path = r'C:\Users\Siddharth Nair\Downloads\signal_space_3d_v3_nli_normal_0.75.png'
        plt.savefig(plot_3d_path, dpi=150, bbox_inches='tight')
        print(f"✓ 3D signal space plot saved to {plot_3d_path}")
        plt.close()
        
        # NLI coherence distribution comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(S3_train_coherence_aug[:n_orig], bins=50, alpha=0.7, 
                    color='purple', edgecolor='black', label='Train')
        axes[0].hist(S3_test_coherence, bins=50, alpha=0.5, 
                    color='orange', edgecolor='black', label='Test')
        axes[0].set_title('NLI Entailment Probability Distribution')
        axes[0].set_xlabel('Entailment Probability')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        
        # Box plot comparison
        data_to_plot = [S3_train_coherence_aug[:n_orig], S3_test_coherence]
        axes[1].boxplot(data_to_plot, labels=['Train', 'Test'])
        axes[1].set_title('NLI Coherence Score Comparison')
        axes[1].set_ylabel('Entailment Probability')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        nli_comparison_path = r'C:\Users\Siddharth Nair\Downloads\nli_coherence_comparison_normal_0.75.png'
        plt.savefig(nli_comparison_path, dpi=150, bbox_inches='tight')
        print(f"✓ NLI coherence comparison saved to {nli_comparison_path}")
        plt.close()
        
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("ENHANCED MODEL (v3) COMPLETE!")
    print("=" * 70)
    print("\nKey Improvements:")
    print("  ✓ S1: Autoencoder-based anomaly detection")
    print("  ✓ S2: Trained Metric-Matcher classifier")
    print("  ✓ S3: NLI-based coherence (MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli)")
    print("       - Uses entailment probability as coherence score")
    print("       - More theoretically sound than cross-encoder scoring")
    print("  ✓ Logically consistent synthetic augmentation")
    print("  ✓ Robust ensemble meta-modeling with Ridge stacking")
    print("\nOutputs:")
    print(f"  1. Submission: {output_path}")
    print(f"  2. Analysis: {analysis_path}")
    print(f"  3. Plots: {plot_path}")
    print(f"  4. 3D Plot: {plot_3d_path}")
    print(f"  5. NLI Comparison: {nli_comparison_path}")
    print("=" * 70)
    
    return ensemble, autoencoder, metric_matcher, nli_scorer, analysis_df


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the complete training and prediction pipeline
    ensemble, autoencoder, metric_matcher, nli_scorer, analysis = main_enhanced_workflow()
    
    print("\n" + "=" * 70)
    print("Pipeline execution completed successfully!")
    print("All models trained and predictions generated.")
    print("=" * 70)